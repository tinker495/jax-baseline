from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.Module import BatchReNorm

SIGMA_INIT = 0.5


class Dense(nn.Dense):
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()


class NoisyDense(nn.Dense):
    rng_collection: str = "params"
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        input_size = jnp.shape(inputs)[-1]
        kernel_mu = self.param(
            "kernel_mu",
            self.kernel_init,
            (input_size, self.features),
            self.param_dtype,
        )
        kernel_sigma = self.param(
            "kernel_sigma",
            jax.nn.initializers.constant(SIGMA_INIT / jnp.sqrt(input_size)),
            (input_size, self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias_mu = self.param("bias_mu", self.bias_init, (self.features,), self.param_dtype)
            bias_sigma = self.param(
                "bias_sigma",
                jax.nn.initializers.constant(SIGMA_INIT / jnp.sqrt(input_size)),
                (self.features,),
                self.param_dtype,
            )
        else:
            bias_mu = None
            bias_sigma = None
        inputs, kernel_mu, kernel_sigma, bias_mu, bias_sigma = promote_dtype(
            inputs, kernel_mu, kernel_sigma, bias_mu, bias_sigma, dtype=self.dtype
        )

        eps_in = self.get_eps(input_size)
        eps_out = self.get_eps(self.features)
        eps_ij = jnp.outer(eps_in, eps_out)
        kernel = kernel_mu + kernel_sigma * eps_ij

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = jax.lax.dot_general

        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias_mu is not None:
            bias = bias_mu + bias_sigma * eps_out
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

    def get_eps(self, n):
        key = self.make_rng(self.rng_collection)
        x = jax.random.normal(key, (n,), dtype=jnp.float32)
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))


# BRO
class BRONet(nn.Module):
    features: int
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = Dense(self.features, kernel_init=self.kernel_init, bias_init=self.bias_init)(inputs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = Dense(self.features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.LayerNorm()(x)
        return x + inputs


# Simba
class ResidualBlock(nn.Module):
    features: int
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()
    middle_feature_multiplier: int = 4
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(inputs)
        x = Dense(
            self.middle_feature_multiplier * self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = self.activation(x)
        x = Dense(self.features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x + inputs


# Simba + BatchNorm
class ResidualBlockBN(nn.Module):
    features: int
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()
    middle_feature_multiplier: int = 4
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.BatchNorm(use_running_average=not training)(inputs)
        x = Dense(
            self.middle_feature_multiplier * self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = self.activation(x)
        x = Dense(self.features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x + inputs


# Simba + BatchRenorm
class ResidualBlockBRN(nn.Module):
    features: int
    kernel_init: Callable = clip_factorized_uniform()
    bias_init: Callable = clip_factorized_uniform()
    middle_feature_multiplier: int = 4
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = BatchReNorm(use_running_average=not training)(inputs)
        x = Dense(
            self.middle_feature_multiplier * self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = self.activation(x)
        x = Dense(self.features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x + inputs


# SimbaV2
def _safe_norm(
    x: jnp.ndarray, axis: int = -1, keepdims: bool = True, eps: float = 1e-6
) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


def l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    return x / _safe_norm(x, axis=axis, keepdims=True, eps=eps)


class Shift(nn.Module):
    c_shift: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype) * self.c_shift
        return jnp.concatenate([inputs, new_axis], axis=-1)


class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        param = self.param(
            "scale",
            nn.initializers.constant(self.scale),
            (self.dim,),
        )
        return param * (self.init / self.scale) * inputs


class LERP(nn.Module):
    features: int
    eps: float = 1e-6
    alpha_init: float = 0.5
    alpha_scale: float = 1.0

    @nn.compact
    def __call__(self, start: jnp.ndarray, end: jnp.ndarray) -> jnp.ndarray:
        alpha_param = self.param(
            "alpha",
            lambda key, shape: jnp.full(
                shape, self.alpha_init * self.alpha_scale, dtype=start.dtype
            ),
            (self.features,),
        )
        alpha = jax.nn.sigmoid(alpha_param)
        blended = (1.0 - alpha) * start + alpha * end
        return l2_normalize(blended, eps=self.eps)


class SimbaV2Embedding(nn.Module):
    hidden_dim: int
    c_shift: float = 3.0
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = Shift(self.c_shift)(inputs)
        x = l2_normalize(x, axis=-1)
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init)(x)
        x = Scaler(self.hidden_dim, init=self.scaler_init, scale=self.scaler_scale)(x)
        x = l2_normalize(x, axis=-1)
        return x


class SimbaV2Block(nn.Module):
    hidden_dim: int
    hidden_multiplier: int = 4
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    alpha_init: float = 0.5
    alpha_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()
    eps: float = 1e-6

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim * self.hidden_multiplier,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(inputs)
        x = Scaler(
            self.hidden_dim * self.hidden_multiplier,
            init=self.scaler_init,
            scale=self.scaler_scale,
        )(x)
        x = nn.relu(x) + self.eps
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init)(x)
        x = l2_normalize(x, axis=-1)
        x = LERP(
            self.hidden_dim,
            eps=self.eps,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
        )(inputs, x)
        return x


class SimbaV2Head(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    kernel_init: Callable = nn.initializers.orthogonal()
    use_bias: bool = False
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.kernel_init)(inputs)
        x = Scaler(self.hidden_dim, init=self.scaler_init, scale=self.scaler_scale)(x)
        x = l2_normalize(x, axis=-1)
        x = nn.Dense(
            self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return x
