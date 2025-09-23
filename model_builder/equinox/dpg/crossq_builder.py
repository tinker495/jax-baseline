from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from model_builder.equinox.Module import (
    BatchReNorm,
    Dense,
    PreProcess,
    extract_batch_stats,
    has_batch_stats,
)
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.equinox.initializers import clip_factorized_uniform
from model_builder.utils import print_param

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN) / 2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN) / 2.0


def _package_params(params_tree):
    batch_stats = extract_batch_stats(params_tree)
    packaged = {"params": params_tree}
    if has_batch_stats(batch_stats):
        packaged["batch_stats"] = batch_stats
    return packaged


class _BRNLayer(eqx.Module):
    dense: Dense
    norm: BatchReNorm
    activation: callable

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.random.KeyArray, activation) -> None:
        k1, _ = jax.random.split(key)
        self.dense = Dense(in_dim, out_dim, key=k1)
        self.norm = BatchReNorm(out_dim)
        self.activation = activation

    def __call__(self, inputs: jnp.ndarray, training: bool = True, *, key: jax.random.KeyArray | None = None):
        x = self.dense(inputs)
        x, new_norm = self.norm(x, training=training)
        x = self.activation(x)
        return x, replace(self, norm=new_norm)


class CrossQActor(eqx.Module):
    pre_norm: BatchReNorm
    layers: Tuple[_BRNLayer, ...]
    mu_head: Dense
    log_std_head: Dense

    def __init__(
        self,
        feature_dim: int,
        action_size: Tuple[int, ...],
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n + 2)
        self.pre_norm = BatchReNorm(feature_dim)
        layers = []
        in_dim = feature_dim
        for i in range(hidden_n):
            layer = _BRNLayer(in_dim, node, key=keys[i], activation=jax.nn.relu)
            layers.append(layer)
            in_dim = node
        self.layers = tuple(layers)
        self.mu_head = Dense(in_dim, action_size[0], key=keys[-2], kernel_init=clip_factorized_uniform(3.0))
        self.log_std_head = Dense(
            in_dim,
            action_size[0],
            key=keys[-1],
            kernel_init=clip_factorized_uniform(3.0),
            bias_init=lambda k, shape, dtype=jnp.float32: jnp.full(shape, 10.0, dtype=dtype),
        )

    def __call__(
        self,
        feature: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        x, new_pre = self.pre_norm(feature, training=training)
        new_layers = []
        for layer in self.layers:
            x, layer = layer(x, training=training)
            new_layers.append(layer)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = LOG_STD_MEAN + LOG_STD_SCALE * jnp.tanh(log_std / LOG_STD_SCALE)
        updated = eqx.tree_at(lambda m: m.pre_norm, self, new_pre)
        updated = eqx.tree_at(lambda m: m.layers, updated, tuple(new_layers))
        return (mu, log_std), updated


class CrossQCritic(eqx.Module):
    pre_norm: BatchReNorm
    action_norm: BatchReNorm
    layers: Tuple[_BRNLayer, ...]
    head: Dense

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n + 1)
        self.pre_norm = BatchReNorm(feature_dim)
        self.action_norm = BatchReNorm(action_dim)
        layers = []
        in_dim = feature_dim + action_dim
        for i in range(hidden_n):
            layer = _BRNLayer(in_dim, node * 8, key=keys[i], activation=jax.nn.tanh)
            layers.append(layer)
            in_dim = node * 8
        self.layers = tuple(layers)
        self.head = Dense(in_dim, 1, key=keys[-1], kernel_init=clip_factorized_uniform(3.0))

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        feat, new_pre = self.pre_norm(feature, training=training)
        act, new_act = self.action_norm(action, training=training)
        x = jnp.concatenate([feat, act], axis=1)
        new_layers = []
        for layer in self.layers:
            x, layer = layer(x, training=training)
            new_layers.append(layer)
        q = self.head(x)
        updated = eqx.tree_at(lambda m: m.pre_norm, self, new_pre)
        updated = eqx.tree_at(lambda m: m.action_norm, updated, new_act)
        updated = eqx.tree_at(lambda m: m.layers, updated, tuple(new_layers))
        return q, updated


class ActorWrapper(eqx.Module):
    preproc: PreProcess
    actor: CrossQActor

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(
        self,
        feature: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        return self.actor(feature, training=training, key=key)


class CriticWrapper(eqx.Module):
    critic1: CrossQCritic
    critic2: CrossQCritic

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        q1, new_c1 = self.critic1(feature, action, training=training, key=key)
        q2, new_c2 = self.critic2(feature, action, training=training, key=key)
        updated = eqx.tree_at(lambda m: m.critic1, self, new_c1)
        updated = eqx.tree_at(lambda m: m.critic2, updated, new_c2)
        return (q1, q2), updated


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_actor, key_critic1, key_critic2 = jax.random.split(rng, 3)

        key_pre, key_actor_body = jax.random.split(key_actor)
        preproc_module = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_dim = preproc_module(dummy_obs).shape[-1]
        actor_core = CrossQActor(feature_dim, action_size, node, hidden_n, key=key_actor_body)
        actor_model = ActorWrapper(preproc_module, actor_core)
        actor_params_tree, actor_static = eqx.partition(actor_model, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        preproc_fn = get_apply_fn_equinox_module(actor_static, actor_model.preprocess)
        actor_fn = get_apply_fn_equinox_module(actor_static, actor_model.actor_forward)

        feature_sample = preproc_fn(actor_params, None, dummy_obs)

        critic_model = CriticWrapper(
            CrossQCritic(feature_sample.shape[-1], action_size[0], node, hidden_n, key=key_critic1),
            CrossQCritic(feature_sample.shape[-1], action_size[0], node, hidden_n, key=key_critic2),
        )
        critic_params_tree, critic_static = eqx.partition(critic_model, eqx.is_array)
        critic_params = _package_params(critic_params_tree)
        critic_fn = get_apply_fn_equinox_module(critic_static, critic_model.__call__)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("actor", actor_params_tree)
                print_param("critic", critic_params_tree)
                print("---------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, actor_params, critic_params
        return preproc_fn, actor_fn, critic_fn

    return model_builder
