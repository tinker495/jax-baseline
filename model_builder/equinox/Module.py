from __future__ import annotations

from dataclasses import field, replace
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from model_builder.equinox.initializers import clip_factorized_uniform

Array = jnp.ndarray


class Dense(eqx.Module):
    weight: Array
    bias: Array | None
    in_features: int = field(static=True)
    out_features: int = field(static=True)
    use_bias: bool = field(static=True)
    kernel_init: Callable = field(static=True, default=clip_factorized_uniform())
    bias_init: Callable = field(static=True, default=clip_factorized_uniform())

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.random.KeyArray,
        use_bias: bool = True,
        kernel_init: Callable = clip_factorized_uniform(),
        bias_init: Callable = clip_factorized_uniform(),
    ) -> None:
        w_key, b_key = jax.random.split(key)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.weight = kernel_init(w_key, (in_features, out_features))
        if use_bias:
            self.bias = bias_init(b_key, (out_features,))
        else:
            self.bias = None

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        out = inputs @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class NoisyDense(eqx.Module):
    mu: Dense
    sigma: Dense
    rng_collection: str = field(static=True, default="params")
    sigma_scale: float = field(static=True, default=0.5)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.random.KeyArray,
        kernel_init: Callable = clip_factorized_uniform(),
        bias_init: Callable = clip_factorized_uniform(),
        sigma_scale: float = 0.5,
    ) -> None:
        mu_key, sigma_key = jax.random.split(key)
        self.mu = Dense(
            in_features,
            out_features,
            key=mu_key,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        sigma_init = lambda k, shape, dtype=jnp.float32: jax.random.uniform(
            k,
            shape,
            dtype,
            -sigma_scale / jnp.sqrt(shape[0]),
            sigma_scale / jnp.sqrt(shape[0]),
        )
        self.sigma = Dense(
            in_features,
            out_features,
            key=sigma_key,
            kernel_init=sigma_init,
            bias_init=sigma_init,
        )
        self.sigma_scale = sigma_scale

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None) -> Array:
        if key is None:
            return self.mu(inputs, key=None)
        in_noise, out_noise = jax.random.split(key)
        eps_in = jax.random.normal(in_noise, (self.mu.in_features,))
        eps_in = jnp.sign(eps_in) * jnp.sqrt(jnp.abs(eps_in))
        eps_out = jax.random.normal(out_noise, (self.mu.out_features,))
        eps_out = jnp.sign(eps_out) * jnp.sqrt(jnp.abs(eps_out))
        w = self.mu.weight + self.sigma.weight * jnp.outer(eps_in, eps_out)
        out = inputs @ w
        if self.mu.bias is not None and self.sigma.bias is not None:
            b = self.mu.bias + self.sigma.bias * eps_out
            out = out + b
        return out


class Flatten(eqx.Module):
    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return inputs.reshape((inputs.shape[0], -1))


class Conv2d(eqx.Module):
    weight: Array
    bias: Array | None
    strides: Tuple[int, int]
    padding: str

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        *,
        key: jax.random.KeyArray,
        strides: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        kernel_init: Callable = clip_factorized_uniform(1.0),
        bias_init: Callable = clip_factorized_uniform(1.0),
    ) -> None:
        weight_key, bias_key = jax.random.split(key)
        self.weight = kernel_init(
            weight_key, (kernel_size[0], kernel_size[1], in_channels, out_channels)
        )
        self.bias = bias_init(bias_key, (out_channels,))
        self.strides = strides
        if padding not in {"SAME", "VALID"}:
            raise ValueError("padding must be 'SAME' or 'VALID'")
        self.padding = padding

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        y = jax.lax.conv_general_dilated(
            inputs,
            self.weight,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        return y


class MaxPool2d(eqx.Module):
    window_shape: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str

    def __init__(
        self,
        window_shape: Tuple[int, int] = (2, 2),
        strides: Tuple[int, int] = (2, 2),
        padding: str = "VALID",
    ) -> None:
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return jax.lax.reduce_window(
            inputs,
            -jnp.inf,
            jax.lax.max,
            (1, *self.window_shape, 1),
            (1, *self.strides, 1),
            self.padding,
        )


def _activation_layers(activation: Callable[[Array], Array]) -> Callable[[Array], Array]:
    def act(x: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return activation(x)

    return act


class Sequential(eqx.Module):
    layers: Tuple

    def __init__(self, layers: Iterable) -> None:
        self.layers = tuple(layers)

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        current = inputs
        layer_key = key
        for layer in self.layers:
            if isinstance(layer, Callable):
                current = layer(current)
            else:
                subkey = None
                if layer_key is not None:
                    layer_key, subkey = jax.random.split(layer_key)
                current = layer(current, key=subkey)
        return current


class VisualEmbedding(eqx.Module):
    network: Sequential

    def __init__(
        self,
        state_shape: Sequence[int],
        *,
        mode: str = "normal",
        flatten: bool = True,
        multiple: int = 1,
        key: jax.random.KeyArray,
    ) -> None:
        h, w, c = state_shape
        keys = iter(jax.random.split(key, 10))
        layers = []
        if mode == "resnet":
            filters = [16 * multiple, 32 * multiple, 32 * multiple]
            in_channels = c
            for f in filters:
                layers.append(
                    Conv2d(
                        in_channels,
                        f,
                        (3, 3),
                        key=next(keys),
                        strides=(1, 1),
                        padding="SAME",
                    )
                )
                layers.append(_activation_layers(jax.nn.relu))
                layers.append(MaxPool2d(window_shape=(3, 3), strides=(2, 2), padding="SAME"))
                in_channels = f
        elif mode == "simple":
            specs = [
                (16, (8, 8), (4, 4), "VALID"),
                (32, (4, 4), (2, 2), "VALID"),
            ]
            in_channels = c
            for out_channels, kernel, stride, padding in specs:
                layers.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        key=next(keys),
                        strides=stride,
                        padding=padding,
                    )
                )
                layers.append(_activation_layers(jax.nn.relu))
                in_channels = out_channels
        elif mode == "minimum":
            specs = [
                (16, (3, 3), (1, 1), "VALID"),
                (32, (4, 4), (2, 2), "VALID"),
            ]
            in_channels = c
            for out_channels, kernel, stride, padding in specs:
                layers.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        key=next(keys),
                        strides=stride,
                        padding=padding,
                    )
                )
                layers.append(_activation_layers(jax.nn.relu))
                in_channels = out_channels
        else:
            specs = [
                (32, (8, 8), (4, 4), "VALID"),
                (64, (4, 4), (2, 2), "VALID"),
                (64, (3, 3), (1, 1), "VALID"),
            ]
            in_channels = c
            for out_channels, kernel, stride, padding in specs:
                layers.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        key=next(keys),
                        strides=stride,
                        padding=padding,
                    )
                )
                layers.append(_activation_layers(jax.nn.relu))
                in_channels = out_channels
        if flatten:
            layers.append(Flatten())
        self.network = Sequential(layers)

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return self.network(inputs, key=key)


class IdentityEmbedding(eqx.Module):
    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return inputs.reshape((inputs.shape[0], -1))


class PreProcess(eqx.Module):
    embeddings: Tuple[eqx.Module, ...]
    pre_postprocess: Callable[[Array], Array] = field(static=True, default=lambda x: x)

    def __init__(
        self,
        states_size: List[Tuple[int, ...]],
        *,
        embedding_mode: str = "normal",
        flatten: bool = True,
        pre_postprocess: Callable[[Array], Array] | None = None,
        multiple: int = 1,
        key: jax.random.KeyArray,
    ) -> None:
        keys = iter(jax.random.split(key, len(states_size) + 1))
        modules = []
        self.output_dim = 0
        for state in states_size:
            if len(state) == 3:
                module = VisualEmbedding(
                    state,
                    mode=embedding_mode,
                    flatten=flatten,
                    multiple=multiple,
                    key=next(keys),
                )
                zero = jnp.zeros((1, *state), dtype=jnp.float32)
                out = module(zero)
                self.output_dim += out.shape[-1]
            else:
                module = IdentityEmbedding()
                zero = jnp.zeros((1, *state), dtype=jnp.float32)
                out = module(zero)
                self.output_dim += out.shape[-1]
            modules.append(module)
        self.embeddings = tuple(modules)
        self.pre_postprocess = pre_postprocess or (lambda x: x)

    def __call__(self, obses: Sequence[Array], *, key: jax.random.KeyArray | None = None) -> Array:
        outputs = []
        key_iter = None
        if key is not None:
            key_iter = iter(jax.random.split(key, len(self.embeddings)))
        for idx, (embed, obs) in enumerate(zip(self.embeddings, obses)):
            subkey = None
            if key_iter is not None:
                subkey = next(key_iter)
            outputs.append(embed(obs, key=subkey))
        concatenated = jnp.concatenate(outputs, axis=1)
        return self.pre_postprocess(concatenated)

    @property
    def output_size(self) -> int:
        return self.output_dim


def sequential_dense(
    in_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    *,
    key: jax.random.KeyArray,
    layer_ctor: Callable[[int, int, jax.random.KeyArray], eqx.Module] | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    last_dim: int | None = None,
    last_activation: Callable[[Array], Array] | None = None,
    kernel_init: Callable = clip_factorized_uniform(1.0),
) -> Tuple[Sequential, int]:
    keys = iter(jax.random.split(key, hidden_layers + (1 if last_dim is not None else 0)))
    layers = []
    current_dim = in_dim
    ctor = layer_ctor or (lambda in_f, out_f, k: Dense(in_f, out_f, key=k, kernel_init=kernel_init, bias_init=kernel_init))
    for _ in range(hidden_layers):
        layer = ctor(current_dim, hidden_dim, next(keys))
        layers.append(layer)
        layers.append(_activation_layers(activation))
        current_dim = hidden_dim
    if last_dim is not None:
        layer = ctor(current_dim, last_dim, next(keys))
        layers.append(layer)
        if last_activation is not None:
            layers.append(_activation_layers(last_activation))
        current_dim = last_dim
    return Sequential(layers), current_dim


class LayerNorm(eqx.Module):
    scale: Array
    bias: Array
    epsilon: float

    def __init__(self, features: int, *, key: jax.random.KeyArray | None = None, epsilon: float = 1e-6):
        key = key or jax.random.PRNGKey(0)
        k_scale, k_bias = jax.random.split(key)
        self.scale = jnp.ones((features,), dtype=jnp.float32)
        self.bias = jnp.zeros((features,), dtype=jnp.float32)
        self.epsilon = epsilon

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.mean((inputs - mean) ** 2, axis=-1, keepdims=True)
        normalized = (inputs - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return normalized * self.scale + self.bias


class BatchStats(eqx.Module):
    mean: Array
    var: Array


class BatchNorm(eqx.Module):
    scale: Array
    bias: Array
    batch_stats: BatchStats
    momentum: float
    epsilon: float

    def __init__(
        self,
        features: int,
        *,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        self.scale = jnp.ones((features,), dtype=jnp.float32)
        self.bias = jnp.zeros((features,), dtype=jnp.float32)
        self.batch_stats = BatchStats(
            mean=jnp.zeros((features,), dtype=jnp.float32),
            var=jnp.ones((features,), dtype=jnp.float32),
        )
        self.momentum = momentum
        self.epsilon = epsilon

    def __call__(
        self,
        inputs: Array,
        *,
        key: jax.random.KeyArray | None = None,
        training: bool = True,
    ) -> tuple[Array, "BatchNorm"]:
        if training:
            batch_mean = jnp.mean(inputs, axis=0)
            batch_var = jnp.var(inputs, axis=0)
            normalized = (inputs - batch_mean) * jax.lax.rsqrt(batch_var + self.epsilon)
            normalized = normalized * self.scale + self.bias
            new_stats = BatchStats(
                mean=self.momentum * self.batch_stats.mean + (1 - self.momentum) * batch_mean,
                var=self.momentum * self.batch_stats.var + (1 - self.momentum) * batch_var,
            )
            updated = replace(self, batch_stats=new_stats)
            return normalized, updated
        normalized = (inputs - self.batch_stats.mean) * jax.lax.rsqrt(
            self.batch_stats.var + self.epsilon
        )
        normalized = normalized * self.scale + self.bias
        return normalized, self


class BatchReNorm(eqx.Module):
    scale: Array
    bias: Array
    batch_stats: BatchStats
    momentum: float
    epsilon: float
    r_max: float
    d_max: float

    def __init__(
        self,
        features: int,
        *,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        r_max: float = 3.0,
        d_max: float = 5.0,
    ) -> None:
        self.scale = jnp.ones((features,), dtype=jnp.float32)
        self.bias = jnp.zeros((features,), dtype=jnp.float32)
        self.batch_stats = BatchStats(
            mean=jnp.zeros((features,), dtype=jnp.float32),
            var=jnp.ones((features,), dtype=jnp.float32),
        )
        self.momentum = momentum
        self.epsilon = epsilon
        self.r_max = r_max
        self.d_max = d_max

    def __call__(
        self,
        inputs: Array,
        *,
        key: jax.random.KeyArray | None = None,
        training: bool = True,
    ) -> tuple[Array, "BatchReNorm"]:
        if training:
            batch_mean = jnp.mean(inputs, axis=0)
            batch_var = jnp.var(inputs, axis=0)
            batch_std = jnp.sqrt(batch_var + self.epsilon)
            running_std = jnp.sqrt(self.batch_stats.var + self.epsilon)
            r = jnp.clip(batch_std / running_std, 1.0 / self.r_max, self.r_max)
            d = jnp.clip((batch_mean - self.batch_stats.mean) / running_std, -self.d_max, self.d_max)
            adjusted_mean = batch_mean - (batch_std * d / r)
            adjusted_var = (batch_var + self.epsilon) / (r * r) - self.epsilon
            normalized = (inputs - adjusted_mean) * jax.lax.rsqrt(adjusted_var + self.epsilon)
            normalized = normalized * self.scale + self.bias
            new_stats = BatchStats(
                mean=self.momentum * self.batch_stats.mean + (1 - self.momentum) * batch_mean,
                var=self.momentum * self.batch_stats.var + (1 - self.momentum) * batch_var,
            )
            updated = replace(self, batch_stats=new_stats)
            return normalized, updated
        normalized = (inputs - self.batch_stats.mean) * jax.lax.rsqrt(
            self.batch_stats.var + self.epsilon
        )
        normalized = normalized * self.scale + self.bias
        return normalized, self


class GroupNorm(eqx.Module):
    scale: Array
    bias: Array
    num_groups: int
    epsilon: float

    def __init__(
        self,
        num_channels: int,
        *,
        num_groups: int = 32,
        epsilon: float = 1e-5,
    ) -> None:
        self.num_groups = min(num_groups, num_channels)
        self.scale = jnp.ones((num_channels,), dtype=jnp.float32)
        self.bias = jnp.zeros((num_channels,), dtype=jnp.float32)
        self.epsilon = epsilon

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        original_shape = inputs.shape
        C = original_shape[-1]
        G = self.num_groups
        if C % G != 0:
            raise ValueError("Number of channels must be divisible by num_groups")
        group_size = C // G
        reshaped = jnp.reshape(inputs, (-1, G, group_size))
        mean = jnp.mean(reshaped, axis=(-1), keepdims=True)
        var = jnp.var(reshaped, axis=(-1), keepdims=True)
        normalized = (reshaped - mean) * jax.lax.rsqrt(var + self.epsilon)
        normalized = jnp.reshape(normalized, original_shape)
        return normalized * self.scale + self.bias


class BRONet(eqx.Module):
    features: int
    dense1: Dense
    dense2: Dense
    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(self, features: int, *, key: jax.random.KeyArray) -> None:
        key1, key2 = jax.random.split(key)
        self.features = features
        self.dense1 = Dense(features, features, key=key1)
        self.dense2 = Dense(features, features, key=key2)
        self.norm1 = LayerNorm(features)
        self.norm2 = LayerNorm(features)

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = self.norm2(x)
        return x + inputs


class ResidualBlock(eqx.Module):
    dense1: Dense
    dense2: Dense
    norm1: LayerNorm
    norm2: LayerNorm
    activation: Callable[[Array], Array]

    def __init__(
        self,
        features: int,
        *,
        key: jax.random.KeyArray,
        middle_feature_multiplier: int = 4,
        activation: Callable[[Array], Array] = jax.nn.relu,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.dense1 = Dense(features, middle_feature_multiplier * features, key=key1)
        self.dense2 = Dense(middle_feature_multiplier * features, features, key=key2)
        self.norm1 = LayerNorm(middle_feature_multiplier * features)
        self.norm2 = LayerNorm(features)
        self.activation = activation

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        x = self.norm1(self.dense1(inputs))
        x = self.activation(x)
        x = self.norm2(self.dense2(x))
        return x + inputs


class ResidualBlockBN(eqx.Module):
    dense1: Dense
    dense2: Dense
    norm1: BatchNorm
    norm2: BatchNorm
    activation: Callable[[Array], Array]

    def __init__(
        self,
        features: int,
        *,
        key: jax.random.KeyArray,
        middle_feature_multiplier: int = 4,
        activation: Callable[[Array], Array] = jax.nn.relu,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.dense1 = Dense(features, middle_feature_multiplier * features, key=key1)
        self.dense2 = Dense(middle_feature_multiplier * features, features, key=key2)
        self.norm1 = BatchNorm(middle_feature_multiplier * features)
        self.norm2 = BatchNorm(features)
        self.activation = activation

    def __call__(
        self,
        inputs: Array,
        *,
        key: jax.random.KeyArray | None = None,
        training: bool = True,
    ) -> tuple[Array, "ResidualBlockBN"]:
        x, norm1 = self.norm1(self.dense1(inputs), training=training)
        x = self.activation(x)
        x, norm2 = self.norm2(self.dense2(x), training=training)
        out = x + inputs
        updated = replace(self, norm1=norm1, norm2=norm2)
        return out, updated


class ResidualBlockBRN(eqx.Module):
    dense1: Dense
    dense2: Dense
    norm1: BatchReNorm
    norm2: BatchReNorm
    activation: Callable[[Array], Array]

    def __init__(
        self,
        features: int,
        *,
        key: jax.random.KeyArray,
        middle_feature_multiplier: int = 4,
        activation: Callable[[Array], Array] = jax.nn.relu,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.dense1 = Dense(features, middle_feature_multiplier * features, key=key1)
        self.dense2 = Dense(middle_feature_multiplier * features, features, key=key2)
        self.norm1 = BatchReNorm(middle_feature_multiplier * features)
        self.norm2 = BatchReNorm(features)
        self.activation = activation

    def __call__(
        self,
        inputs: Array,
        *,
        key: jax.random.KeyArray | None = None,
        training: bool = True,
    ) -> tuple[Array, "ResidualBlockBRN"]:
        x, norm1 = self.norm1(self.dense1(inputs), training=training)
        x = self.activation(x)
        x, norm2 = self.norm2(self.dense2(x), training=training)
        out = x + inputs
        updated = replace(self, norm1=norm1, norm2=norm2)
        return out, updated


def _safe_norm(x: Array, axis: int = -1, keepdims: bool = True, eps: float = 1e-6) -> Array:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    return x / _safe_norm(x, axis=axis, keepdims=True, eps=eps)


class Shift(eqx.Module):
    c_shift: float = 1.0

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        new_axis = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype) * self.c_shift
        return jnp.concatenate([inputs, new_axis], axis=-1)


class Scaler(eqx.Module):
    scale: Array

    def __init__(self, dim: int, *, init: float = 1.0) -> None:
        self.scale = jnp.ones((dim,), dtype=jnp.float32) * init

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return inputs * self.scale


class LERP(eqx.Module):
    alpha: Array

    def __init__(self, features: int, *, alpha_init: float = 0.5) -> None:
        self.alpha = jnp.ones((features,), dtype=jnp.float32) * alpha_init

    def __call__(self, start: Array, end: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        return start + self.alpha * (end - start)


class SimbaV2Embedding(eqx.Module):
    dense: Dense
    scaler: Scaler
    c_shift: float

    def __init__(
        self,
        input_dim: int,
        *,
        key: jax.random.KeyArray,
        hidden_dim: int,
        c_shift: float = 3.0,
        scaler_init: float = 1.0,
    ) -> None:
        k_dense, _ = jax.random.split(key)
        self.dense = Dense(input_dim + 1, hidden_dim, key=k_dense, use_bias=False)
        self.scaler = Scaler(hidden_dim, init=scaler_init)
        self.c_shift = c_shift

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        x = Shift(self.c_shift)(inputs)
        x = l2_normalize(x, axis=-1)
        x = self.dense(x)
        x = self.scaler(x)
        return l2_normalize(x, axis=-1)


class SimbaV2Block(eqx.Module):
    dense1: Dense
    scaler: Scaler
    dense2: Dense
    lerp: LERP

    def __init__(
        self,
        hidden_dim: int,
        *,
        key: jax.random.KeyArray,
        hidden_multiplier: int = 4,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 0.5,
        alpha_scale: float = 1.0,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.dense1 = Dense(hidden_dim, hidden_multiplier * hidden_dim, key=k1, use_bias=False)
        self.scaler = Scaler(hidden_multiplier * hidden_dim, init=scaler_init)
        self.dense2 = Dense(hidden_multiplier * hidden_dim, hidden_dim, key=k2, use_bias=False)
        self.lerp = LERP(hidden_dim, alpha_init=alpha_init)

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        x = self.dense1(inputs)
        x = self.scaler(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = l2_normalize(x, axis=-1)
        x = self.lerp(inputs, x)
        return l2_normalize(x, axis=-1)


class SimbaV2Head(eqx.Module):
    dense1: Dense
    scaler: Scaler
    dense2: Dense

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        *,
        key: jax.random.KeyArray,
        scaler_init: float = 1.0,
        use_bias: bool = False,
        bias_init: Callable | None = None,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.dense1 = Dense(hidden_dim, hidden_dim, key=k1, use_bias=False)
        self.scaler = Scaler(hidden_dim, init=scaler_init)
        bias_init = bias_init or (lambda k, shape, dtype=jnp.float32: jnp.zeros(shape, dtype))
        self.dense2 = Dense(
            hidden_dim,
            out_dim,
            key=k2,
            use_bias=use_bias,
            bias_init=bias_init,
        )

    def __call__(self, inputs: Array, *, key: jax.random.KeyArray | None = None) -> Array:
        x = self.dense1(inputs)
        x = self.scaler(x)
        return self.dense2(x)


def avgl1norm(x: Array, epsilon: float = 1e-6) -> Array:
    return x / (jnp.mean(jnp.abs(x), axis=-1, keepdims=True) + epsilon)


def extract_batch_stats(tree):
    def _extract(node):
        if isinstance(node, BatchStats):
            return {"mean": node.mean, "var": node.var}
        return None

    return jtu.tree_map(_extract, tree, is_leaf=lambda x: isinstance(x, BatchStats))


def apply_batch_stats(module, stats_tree):
    if stats_tree is None:
        return module

    def _apply(node, stats):
        if isinstance(node, BatchStats) and stats is not None:
            return BatchStats(mean=stats["mean"], var=stats["var"])
        return node

    return jtu.tree_map(
        _apply,
        module,
        stats_tree,
        is_leaf=lambda x: isinstance(x, BatchStats) or isinstance(x, dict) or x is None,
    )


def has_batch_stats(stats_tree) -> bool:
    if stats_tree is None:
        return False
    leaves = [leaf for leaf in jtu.tree_leaves(stats_tree) if leaf is not None]
    return len(leaves) > 0
