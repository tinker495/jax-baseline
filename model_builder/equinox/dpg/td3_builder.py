from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import (
    Dense,
    PreProcess,
    Sequential,
    extract_batch_stats,
    has_batch_stats,
    sequential_dense,
)
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param


def _package_params(params_tree):
    batch_stats = extract_batch_stats(params_tree)
    packaged = {"params": params_tree}
    if has_batch_stats(batch_stats):
        packaged["batch_stats"] = batch_stats
    return packaged


class ActorCore(eqx.Module):
    network: Sequential

    def __init__(
        self,
        feature_dim: int,
        action_size: Tuple[int, ...],
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, head_key = jax.random.split(key)
        trunk, hidden_dim = sequential_dense(feature_dim, node, hidden_n, key=trunk_key)
        layers = list(trunk.layers)
        layers.append(Dense(hidden_dim, action_size[0], key=head_key))
        layers.append(lambda x: jnp.tanh(x))
        self.network = Sequential(layers)

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        return self.network(feature, key=key)


class CriticCore(eqx.Module):
    network: Sequential

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, head_key = jax.random.split(key)
        input_dim = feature_dim + action_dim
        trunk, hidden_dim = sequential_dense(input_dim, node, hidden_n, key=trunk_key)
        layers = list(trunk.layers)
        layers.append(Dense(hidden_dim, 1, key=head_key))
        self.network = Sequential(layers)

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, action], axis=1)
        return self.network(concat, key=key)


class MergedActor(eqx.Module):
    preproc: PreProcess
    actor: ActorCore

    def __init__(
        self,
        observation_space,
        action_size,
        *,
        embedding_mode: str,
        node: int,
        hidden_n: int,
        key: jax.random.KeyArray,
    ) -> None:
        key_pre, key_actor = jax.random.split(key)
        self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        feature_dim = self.preproc.output_size
        self.actor = ActorCore(feature_dim, action_size, node, hidden_n, key=key_actor)

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(
        self,
        feature: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        return self.actor(feature, key=key)


class DoubleCritic(eqx.Module):
    critic1: CriticCore
    critic2: CriticCore

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        node: int,
        hidden_n: int,
        key: jax.random.KeyArray,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.critic1 = CriticCore(feature_dim, action_dim, node, hidden_n, key=key1)
        self.critic2 = CriticCore(feature_dim, action_dim, node, hidden_n, key=key2)

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(feature, action, key=key)
        q2 = self.critic2(feature, action, key=key)
        return q1, q2


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_actor, key_critic = jax.random.split(rng)

        actor_model = MergedActor(
            observation_space,
            action_size,
            embedding_mode=embedding_mode,
            node=node,
            hidden_n=hidden_n,
            key=key_actor,
        )
        policy_params_tree, policy_static = eqx.partition(actor_model, eqx.is_array)
        preproc_fn = get_apply_fn_equinox_module(policy_static, actor_model.preprocess)
        actor_fn = get_apply_fn_equinox_module(policy_static, actor_model.actor_forward)
        policy_params = _package_params(policy_params_tree)

        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_sample = preproc_fn(policy_params, None, dummy_obs)

        critic_model = DoubleCritic(
            feature_sample.shape[-1],
            action_size[0],
            node=node,
            hidden_n=hidden_n,
            key=key_critic,
        )
        critic_params_tree, critic_static = eqx.partition(critic_model, eqx.is_array)
        critic_fn = get_apply_fn_equinox_module(critic_static, critic_model.__call__)
        critic_params = _package_params(critic_params_tree)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("policy", policy_params_tree)
                print_param("critic", critic_params_tree)
                print("---------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        return preproc_fn, actor_fn, critic_fn

    return model_builder
