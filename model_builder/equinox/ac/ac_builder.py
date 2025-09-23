from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import Dense, PreProcess, Sequential, sequential_dense
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param


class Actor(eqx.Module):
    action_type: str
    backbone: Sequential
    head: Dense
    log_std: jnp.ndarray | None

    def __init__(
        self,
        feature_dim: int,
        action_size: Tuple[int, ...],
        action_type: str,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, head_key = jax.random.split(key)
        self.action_type = action_type
        self.backbone, hidden_dim = sequential_dense(
            feature_dim,
            node,
            hidden_n,
            key=trunk_key,
        )
        self.head = Dense(hidden_dim, action_size[0], key=head_key)
        if action_type == "continuous":
            self.log_std = jnp.zeros((1, action_size[0]), dtype=jnp.float32)
        else:
            self.log_std = None

    def __call__(
        self,
        feature: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        hidden = self.backbone(feature, key=key)
        logits = self.head(hidden, key=key)
        if self.action_type == "discrete":
            return logits
        mu = jnp.tanh(logits)
        return mu, self.log_std


class Critic(eqx.Module):
    backbone: Sequential
    head: Dense

    def __init__(
        self,
        feature_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, head_key = jax.random.split(key)
        self.backbone, hidden_dim = sequential_dense(
            feature_dim,
            node,
            hidden_n,
            key=trunk_key,
        )
        self.head = Dense(hidden_dim, 1, key=head_key)

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        hidden = self.backbone(feature, key=key)
        value = self.head(hidden, key=key)
        return value


class ActorCritic(eqx.Module):
    preproc: PreProcess
    actor: Actor
    critic: Critic

    def preprocess(
        self,
        obses,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(
        self,
        obses,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        feature = self.preprocess(obses, key=key)
        return self.actor(feature, key=key)

    def critic_forward(
        self,
        obses,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        feature = self.preprocess(obses, key=key)
        return self.critic(feature, key=key)


def model_builder_maker(observation_space, action_size, action_type, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def _model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_actor, key_critic = jax.random.split(rng, 3)
        preproc = PreProcess(
            observation_space,
            embedding_mode=embedding_mode,
            key=key_pre,
        )
        feature_dim = preproc.output_size
        actor = Actor(
            feature_dim,
            action_size,
            action_type,
            node=node,
            hidden_n=hidden_n,
            key=key_actor,
        )
        critic = Critic(
            feature_dim,
            node=node,
            hidden_n=hidden_n,
            key=key_critic,
        )
        merged = ActorCritic(preproc, actor, critic)
        params, static = eqx.partition(merged, eqx.is_array)

        preproc_fn = get_apply_fn_equinox_module(static, merged.preprocess)
        actor_fn = get_apply_fn_equinox_module(static, merged.actor_forward)
        critic_fn = get_apply_fn_equinox_module(static, merged.critic_forward)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("", params)
                print("---------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, params
        return preproc_fn, actor_fn, critic_fn

    return _model_builder
