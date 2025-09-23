from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import (
    Dense,
    LayerNorm,
    PreProcess,
    ResidualBlock,
    avgl1norm,
    extract_batch_stats,
    has_batch_stats,
)
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.equinox.initializers import clip_factorized_uniform
from model_builder.utils import print_param


def _package_params(params_tree):
    batch_stats = extract_batch_stats(params_tree)
    packaged = {"params": params_tree}
    if has_batch_stats(batch_stats):
        packaged["batch_stats"] = batch_stats
    return packaged


class Encoder(eqx.Module):
    layers: tuple[Dense, ...]
    activation: callable

    def __init__(self, feature_dim: int, node: int, hidden_n: int, *, key: jax.random.KeyArray) -> None:
        keys = jax.random.split(key, hidden_n)
        layers = []
        in_dim = feature_dim
        for i in range(hidden_n):
            layers.append(Dense(in_dim, node, key=keys[i]))
            in_dim = node
        self.layers = tuple(layers)
        self.activation = jax.nn.elu

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        x = feature
        for dense in self.layers:
            x = dense(x)
            x = self.activation(x)
        return avgl1norm(x)


class ActionEncoder(eqx.Module):
    layers: tuple[Dense, ...]
    activation: callable

    def __init__(
        self,
        zs_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n)
        layers = []
        in_dim = zs_dim + action_dim
        for i in range(hidden_n):
            layers.append(Dense(in_dim, node, key=keys[i]))
            in_dim = node
        self.layers = tuple(layers)
        self.activation = jax.nn.elu

    def __call__(
        self,
        zs: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        x = jnp.concatenate([zs, actions], axis=1)
        for dense in self.layers:
            x = dense(x)
            x = self.activation(x)
        return x


class SimbaTD7Actor(eqx.Module):
    feature_dense: Dense
    hidden_dense: Dense
    blocks: tuple[ResidualBlock, ...]
    norm: LayerNorm
    head: Dense

    def __init__(
        self,
        feature_dim: int,
        zs_dim: int,
        action_size: tuple[int, ...],
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        key_f, key_rest = jax.random.split(key)
        self.feature_dense = Dense(feature_dim, node, key=key_f)
        keys = jax.random.split(key_rest, hidden_n + 2)
        self.hidden_dense = Dense(node + zs_dim, node, key=keys[0])
        self.blocks = tuple(ResidualBlock(node, key=keys[i + 1]) for i in range(hidden_n))
        self.norm = LayerNorm(node)
        self.head = Dense(
            node,
            action_size[0],
            key=keys[-1],
            kernel_init=clip_factorized_uniform(3.0),
        )

    def __call__(
        self,
        feature: jnp.ndarray,
        zs: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        a0 = avgl1norm(self.feature_dense(feature))
        x = jnp.concatenate([a0, zs], axis=1)
        x = self.hidden_dense(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return jax.nn.tanh(self.head(x))


class SimbaTD7Critic(eqx.Module):
    feature_dense: Dense
    hidden_dense: Dense
    blocks: tuple[ResidualBlock, ...]
    norm: LayerNorm
    head: Dense

    def __init__(
        self,
        feature_dim: int,
        zs_dim: int,
        zsa_dim: int,
        action_dim: int,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        key_f, key_rest = jax.random.split(key)
        self.feature_dense = Dense(feature_dim + action_dim, node, key=key_f)
        keys = jax.random.split(key_rest, hidden_n + 2)
        self.hidden_dense = Dense(node + zs_dim + zsa_dim, node, key=keys[0])
        self.blocks = tuple(ResidualBlock(node, key=keys[i + 1]) for i in range(hidden_n))
        self.norm = LayerNorm(node)
        self.head = Dense(node, 1, key=keys[-1], kernel_init=clip_factorized_uniform(3.0))
        self.activation = jax.nn.elu

    def __call__(
        self,
        feature: jnp.ndarray,
        zs: jnp.ndarray,
        zsa: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        x = jnp.concatenate([feature, actions], axis=1)
        q0 = avgl1norm(self.feature_dense(x))
        x = jnp.concatenate([q0, zs, zsa], axis=1)
        x = self.hidden_dense(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class EncoderWrapper(eqx.Module):
    preproc: PreProcess
    encoder: Encoder
    action_encoder: ActionEncoder

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def encoder_forward(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        return self.encoder(feature, key=key)

    def action_encoder_forward(
        self,
        zs: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        return self.action_encoder(zs, actions, key=key)

    def feature_and_zs(self, obses, *, key: jax.random.KeyArray | None = None):
        feature = self.preprocess(obses, key=key)
        zs = self.encoder(feature, key=key)
        return feature, zs


class DoubleCritic(eqx.Module):
    critic1: SimbaTD7Critic
    critic2: SimbaTD7Critic

    def __call__(
        self,
        feature: jnp.ndarray,
        zs: jnp.ndarray,
        zsa: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        q1 = self.critic1(feature, zs, zsa, action, key=key1)
        q2 = self.critic2(feature, zs, zsa, action, key=key2)
        return q1, q2


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)
    encoder_hidden = policy_kwargs.get("encoder_hidden", 3)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_enc, key_act_enc, key_actor, key_c1, key_c2 = jax.random.split(rng, 6)

        preproc_module = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_dim = preproc_module(dummy_obs).shape[-1]

        encoder_module = EncoderWrapper(
            preproc_module,
            Encoder(feature_dim, node, encoder_hidden, key=key_enc),
            ActionEncoder(node, action_size[0], node, encoder_hidden, key=key_act_enc),
        )
        encoder_params_tree, encoder_static = eqx.partition(encoder_module, eqx.is_array)
        encoder_params = _package_params(encoder_params_tree)
        preproc_fn = get_apply_fn_equinox_module(encoder_static, encoder_module.preprocess)
        encoder_fn = get_apply_fn_equinox_module(encoder_static, encoder_module.encoder_forward)
        action_encoder_fn = get_apply_fn_equinox_module(encoder_static, encoder_module.action_encoder_forward)
        feature_and_zs_fn = get_apply_fn_equinox_module(encoder_static, encoder_module.feature_and_zs)

        feature_sample, zs_sample = feature_and_zs_fn(encoder_params, None, dummy_obs)
        dummy_action = np.zeros((1, action_size[0]), dtype=np.float32)
        zsa_sample = action_encoder_fn(encoder_params, None, zs_sample, dummy_action)

        actor_model = SimbaTD7Actor(feature_dim, zs_sample.shape[-1], action_size, node, hidden_n, key=key_actor)
        critic_model = DoubleCritic(
            SimbaTD7Critic(feature_dim, zs_sample.shape[-1], zsa_sample.shape[-1], action_size[0], node, hidden_n, key=key_c1),
            SimbaTD7Critic(feature_dim, zs_sample.shape[-1], zsa_sample.shape[-1], action_size[0], node, hidden_n, key=key_c2),
        )

        actor_params_tree, actor_static = eqx.partition(actor_model, eqx.is_array)
        critic_params_tree, critic_static = eqx.partition(critic_model, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        critic_params = _package_params(critic_params_tree)
        actor_fn = get_apply_fn_equinox_module(actor_static)
        critic_fn = get_apply_fn_equinox_module(critic_static)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("encoder", encoder_params_tree)
                print_param("actor", actor_params_tree)
                print_param("critic", critic_params_tree)
                print("---------------------------------------------------------")
            return (
                preproc_fn,
                encoder_fn,
                action_encoder_fn,
                actor_fn,
                critic_fn,
                encoder_params,
                actor_params,
                critic_params,
            )
        return preproc_fn, encoder_fn, action_encoder_fn, actor_fn, critic_fn

    return model_builder
