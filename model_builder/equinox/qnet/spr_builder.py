from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import (
    Conv2d,
    Dense,
    GroupNorm,
    PreProcess,
)
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.equinox.qnet.c51_builder import C51Network
from model_builder.utils import print_param


class ConvBlock(eqx.Module):
    conv: Conv2d
    norm: GroupNorm

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.random.KeyArray) -> None:
        k_conv, _ = jax.random.split(key)
        self.conv = Conv2d(in_channels, out_channels, (3, 3), key=k_conv, strides=(1, 1), padding="SAME")
        self.norm = GroupNorm(out_channels, num_groups=4)

    def __call__(self, inputs: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        x = self.conv(inputs)
        x = self.norm(x)
        return jax.nn.relu(x)


class Transition(eqx.Module):
    convs: tuple[ConvBlock, ...]
    last: Conv2d

    def __init__(
        self,
        feature_shape,
        action_dim: int,
        *,
        channels: int = 64,
        hidden_n: int = 2,
        key: jax.random.KeyArray,
    ) -> None:
        keys = jax.random.split(key, hidden_n + 1)
        in_channels = feature_shape[-1] + action_dim
        convs = []
        for i in range(hidden_n):
            convs.append(ConvBlock(in_channels, channels, key=keys[i]))
            in_channels = channels
        self.convs = tuple(convs)
        self.last = Conv2d(channels, feature_shape[-1], (3, 3), key=keys[-1], strides=(1, 1), padding="SAME")

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        batch = feature.shape[0]
        action = jnp.reshape(action, (batch, 1, 1, -1))
        action = jnp.tile(action, (1, feature.shape[1], feature.shape[2], 1))
        x = jnp.concatenate([feature, action], axis=-1)
        for block in self.convs:
            x = block(x)
        return self.last(x)


class Projection(eqx.Module):
    flatten_dim: int
    proj_layers: tuple
    out: Dense
    embed_size: int

    def __init__(self, feature_shape, embed_size: int = 128, *, node: int = 512, hidden_n: int = 2, key: jax.random.KeyArray):
        keys = jax.random.split(key, hidden_n + 1)
        self.flatten_dim = int(np.prod(feature_shape))
        layers = []
        in_dim = self.flatten_dim
        for i in range(hidden_n):
            layers.append(Dense(in_dim, node, key=keys[i]))
            layers.append(lambda x: jax.nn.relu(x))
            in_dim = node
        self.proj_layers = tuple(layers)
        self.out = Dense(in_dim, embed_size, key=keys[-1])
        self.embed_size = embed_size

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        x = jnp.reshape(feature, (feature.shape[0], self.flatten_dim))
        for layer in self.proj_layers:
            if isinstance(layer, eqx.Module):
                x = layer(x)
            else:
                x = layer(x)
        return self.out(x)


class Prediction(eqx.Module):
    layers: tuple

    def __init__(self, embed_dim: int, *, node: int = 128, hidden_n: int = 1, key: jax.random.KeyArray) -> None:
        keys = jax.random.split(key, hidden_n + 1)
        layers = []
        in_dim = embed_dim
        for i in range(hidden_n):
            layers.append(Dense(in_dim, node, key=keys[i]))
            layers.append(lambda x: jax.nn.relu(x))
            in_dim = node
        layers.append(Dense(in_dim, embed_dim, key=keys[-1]))
        self.layers = tuple(layers)

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        x = feature
        for layer in self.layers:
            if isinstance(layer, eqx.Module):
                x = layer(x)
            else:
                x = layer(x)
        return x


class SPR(eqx.Module):
    preproc: PreProcess
    qnet: C51Network
    transition: Transition
    projection: Projection
    prediction: Prediction

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def q(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        flat = jnp.reshape(feature, (feature.shape[0], -1))
        return self.qnet(flat, key=key)

    def transition_forward(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        return self.transition(feature, action, key=key)

    def projection_forward(self, feature: jnp.ndarray) -> jnp.ndarray:
        return self.projection(feature)

    def prediction_forward(self, feature: jnp.ndarray) -> jnp.ndarray:
        return self.prediction(feature)

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ):
        q = self.q(feature, key=key)
        transition = self.transition_forward(feature, action, key=key)
        projection = self.projection_forward(transition)
        prediction = self.prediction_forward(projection)
        return q, transition, projection, prediction


def model_builder_maker(
    observation_space,
    action_space,
    dueling_model,
    param_noise,
    categorial_bar_n,
    policy_kwargs,
):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_q, key_trans, key_proj, key_pred = jax.random.split(rng, 5)
        preproc = PreProcess(
            observation_space,
            embedding_mode=embedding_mode,
            key=key_pre,
            flatten=False,
        )
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_sample = preproc(dummy_obs)
        flat_dim = int(np.prod(feature_sample.shape[1:]))
        qnet = C51Network(
            flat_dim,
            action_space[0],
            categorial_bar_n,
            node=node,
            hidden_n=hidden_n,
            dueling=dueling_model,
            noisy=param_noise,
            key=key_q,
        )
        transition = Transition(feature_sample.shape[1:], action_space[0], key=key_trans)
        projection = Projection(feature_sample.shape[1:], key=key_proj)
        prediction = Prediction(projection.embed_size, key=key_pred)
        model = SPR(preproc, qnet, transition, projection, prediction)
        params, static = eqx.partition(model, eqx.is_array)
        preproc_fn = get_apply_fn_equinox_module(static, model.preprocess)
        q_fn = get_apply_fn_equinox_module(static, model.q)
        transition_fn = get_apply_fn_equinox_module(static, model.transition_forward)
        projection_fn = get_apply_fn_equinox_module(static, model.projection_forward)
        prediction_fn = get_apply_fn_equinox_module(static, model.prediction_forward)
        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("spr", params)
                print("---------------------------------------------------------")
            return preproc_fn, q_fn, transition_fn, projection_fn, prediction_fn, params
        return preproc_fn, q_fn, transition_fn, projection_fn, prediction_fn

    return model_builder
