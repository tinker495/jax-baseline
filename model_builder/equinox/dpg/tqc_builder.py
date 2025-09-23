from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import Dense, PreProcess, extract_batch_stats, has_batch_stats, sequential_dense
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


class ActorCore(eqx.Module):
    backbone: eqx.Module
    mu_head: Dense
    log_std_head: Dense

    def __init__(
        self,
        feature_dim: int,
        action_size: tuple[int, ...],
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, mu_key, log_std_key = jax.random.split(key, 3)
        self.backbone, hidden_dim = sequential_dense(feature_dim, node, hidden_n, key=trunk_key)
        self.mu_head = Dense(hidden_dim, action_size[0], key=mu_key, kernel_init=clip_factorized_uniform(3.0))
        self.log_std_head = Dense(
            hidden_dim,
            action_size[0],
            key=log_std_key,
            kernel_init=clip_factorized_uniform(3.0),
            bias_init=lambda k, shape, dtype=jnp.float32: jnp.full(shape, 10.0, dtype=dtype),
        )

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        hidden = self.backbone(feature, key=key)
        mu = self.mu_head(hidden, key=key)
        log_std = self.log_std_head(hidden, key=key)
        log_std = LOG_STD_MEAN + LOG_STD_SCALE * jnp.tanh(log_std / LOG_STD_SCALE)
        return mu, log_std


class CriticCore(eqx.Module):
    backbone: eqx.Module
    head: Dense

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        support_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        trunk_key, head_key = jax.random.split(key)
        self.backbone, hidden_dim = sequential_dense(feature_dim + action_dim, node, hidden_n, key=trunk_key)
        self.head = Dense(
            hidden_dim,
            support_n,
            key=head_key,
            kernel_init=clip_factorized_uniform(3.0 / support_n),
        )

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, action], axis=1)
        hidden = self.backbone(concat, key=key)
        return self.head(hidden, key=key)


class ActorWrapper(eqx.Module):
    preproc: PreProcess
    actor: ActorCore

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        return self.actor(feature, key=key)


class DoubleCritic(eqx.Module):
    critic1: CriticCore
    critic2: CriticCore

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        q1 = self.critic1(feature, action, key=key1)
        q2 = self.critic2(feature, action, key=key2)
        return q1, q2


def model_builder_maker(observation_space, action_size, support_n, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_actor, key_c1, key_c2 = jax.random.split(rng, 4)

        preproc_module = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_dim = preproc_module(dummy_obs).shape[-1]

        actor_core = ActorCore(feature_dim, action_size, node, hidden_n, key=key_actor)
        actor_wrapper = ActorWrapper(preproc_module, actor_core)
        actor_params_tree, actor_static = eqx.partition(actor_wrapper, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        preproc_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.preprocess)
        actor_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.actor_forward)

        critic1 = CriticCore(feature_dim, action_size[0], node, hidden_n, support_n, key=key_c1)
        critic2 = CriticCore(feature_dim, action_size[0], node, hidden_n, support_n, key=key_c2)
        critic_model = DoubleCritic(critic1, critic2)
        critic_params_tree, critic_static = eqx.partition(critic_model, eqx.is_array)
        critic_params = _package_params(critic_params_tree)
        critic_fn = get_apply_fn_equinox_module(critic_static)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("actor", actor_params_tree)
                print_param("critic", critic_params_tree)
                print("---------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, actor_params, critic_params
        return preproc_fn, actor_fn, critic_fn

    return model_builder
