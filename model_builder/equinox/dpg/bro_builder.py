from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import BRONet, Dense, LayerNorm, PreProcess, extract_batch_stats, has_batch_stats
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


class BroActor(eqx.Module):
    input_dense: Dense
    input_norm: LayerNorm
    blocks: tuple[BRONet, ...]
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
        keys = jax.random.split(key, hidden_n + 3)
        self.input_dense = Dense(feature_dim, node, key=keys[0])
        self.input_norm = LayerNorm(node)
        blocks = []
        for i in range(hidden_n):
            blocks.append(BRONet(node, key=keys[i + 1]))
        self.blocks = tuple(blocks)
        self.mu_head = Dense(node, action_size[0], key=keys[-2], kernel_init=clip_factorized_uniform(3.0))
        self.log_std_head = Dense(
            node,
            action_size[0],
            key=keys[-1],
            kernel_init=clip_factorized_uniform(3.0),
            bias_init=lambda k, shape, dtype=jnp.float32: jnp.full(shape, 10.0, dtype=dtype),
        )

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        x = self.input_dense(feature)
        x = self.input_norm(x)
        x = jax.nn.relu(x)
        for block in self.blocks:
            x = block(x)
        mu_log_std = jnp.concatenate([self.mu_head(x), self.log_std_head(x)], axis=-1)
        mu, log_std = jnp.split(mu_log_std, 2, axis=-1)
        log_std = LOG_STD_MEAN + LOG_STD_SCALE * jnp.tanh(log_std / LOG_STD_SCALE)
        return mu, log_std


class BroCritic(eqx.Module):
    input_dense: Dense
    input_norm: LayerNorm
    blocks: tuple[BRONet, ...]
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
        keys = jax.random.split(key, hidden_n + 2)
        self.input_dense = Dense(feature_dim + action_dim, node, key=keys[0])
        self.input_norm = LayerNorm(node)
        blocks = []
        for i in range(hidden_n):
            blocks.append(BRONet(node, key=keys[i + 1]))
        self.blocks = tuple(blocks)
        self.head = Dense(node, 1, key=keys[-1], kernel_init=clip_factorized_uniform(3.0))

    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        x = jnp.concatenate([feature, actions], axis=1)
        x = self.input_dense(x)
        x = self.input_norm(x)
        x = jax.nn.relu(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ActorWrapper(eqx.Module):
    preproc: PreProcess
    actor: BroActor

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        return self.actor(feature, key=key)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_actor, key_critic = jax.random.split(rng, 3)

        preproc_module = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_dim = preproc_module(dummy_obs).shape[-1]

        actor_core = BroActor(feature_dim, action_size, node, hidden_n, key=key_actor)
        actor_wrapper = ActorWrapper(preproc_module, actor_core)
        actor_params_tree, actor_static = eqx.partition(actor_wrapper, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        preproc_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.preprocess)
        actor_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.actor_forward)

        critic_model = BroCritic(feature_dim, action_size[0], node, hidden_n, key=key_critic)
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
