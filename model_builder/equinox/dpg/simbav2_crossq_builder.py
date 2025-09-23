from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import (
    PreProcess,
    SimbaV2Block,
    SimbaV2Embedding,
    SimbaV2Head,
    extract_batch_stats,
    has_batch_stats,
)
from model_builder.equinox.apply import get_apply_fn_equinox_module
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


class SimbaV2CrossQActor(eqx.Module):
    embedding: SimbaV2Embedding
    blocks: tuple[SimbaV2Block, ...]
    mu_head: SimbaV2Head
    log_std_head: SimbaV2Head

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n + 3)
        self.embedding = SimbaV2Embedding(feature_dim, hidden_dim=node, key=keys[0])
        self.blocks = tuple(SimbaV2Block(node, key=keys[i + 1]) for i in range(hidden_n))
        self.mu_head = SimbaV2Head(node, action_dim, key=keys[-2], use_bias=True)
        self.log_std_head = SimbaV2Head(
            node,
            action_dim,
            key=keys[-1],
            use_bias=True,
            bias_init=lambda k, shape, dtype=jnp.float32: jnp.full(shape, 10.0, dtype=dtype),
        )

    def __call__(
        self,
        feature: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        del training
        x = self.embedding(feature)
        for block in self.blocks:
            x = block(x)
        mu = self.mu_head(x)
        log_std_raw = self.log_std_head(x)
        log_std = LOG_STD_MEAN + LOG_STD_SCALE * jnp.tanh(log_std_raw / LOG_STD_SCALE)
        return mu, log_std


class SimbaV2CrossQCritic(eqx.Module):
    embedding: SimbaV2Embedding
    blocks: tuple[SimbaV2Block, ...]
    head: SimbaV2Head

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n + 2)
        self.embedding = SimbaV2Embedding(feature_dim + action_dim, hidden_dim=node, key=keys[0])
        self.blocks = tuple(SimbaV2Block(node, key=keys[i + 1]) for i in range(hidden_n))
        self.head = SimbaV2Head(node, 1, key=keys[-1], use_bias=True)

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        del training
        x = jnp.concatenate([feature, action], axis=1)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ActorWrapper(eqx.Module):
    preproc: PreProcess
    actor: SimbaV2CrossQActor

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


class DoubleCritic(eqx.Module):
    critic1: SimbaV2CrossQCritic
    critic2: SimbaV2CrossQCritic

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = True,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        del training
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        q1 = self.critic1(feature, action, key=key1)
        q2 = self.critic2(feature, action, key=key2)
        return q1, q2


def model_builder_maker(observation_space, action_size, policy_kwargs):
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

        actor_core = SimbaV2CrossQActor(feature_dim, action_size[0], node, hidden_n, key=key_actor)
        actor_wrapper = ActorWrapper(preproc_module, actor_core)
        actor_params_tree, actor_static = eqx.partition(actor_wrapper, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        preproc_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.preprocess)
        actor_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.actor_forward)

        critic_model = DoubleCritic(
            SimbaV2CrossQCritic(feature_dim, action_size[0], node, hidden_n, key=key_c1),
            SimbaV2CrossQCritic(feature_dim, action_size[0], node, hidden_n, key=key_c2),
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
