from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.model_config import ACTIVATIONS, MLPConfig, ModelConfig

SIGMA_INIT = 0.5

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN) / 2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN) / 2.0


def avgl1norm(x: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True) + epsilon)


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


# SimbaV2
def _safe_norm(
    x: jnp.ndarray, axis: int = -1, keepdims: bool = True, eps: float = 1e-6
) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims) + eps)


def l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    return x / _safe_norm(x, axis=axis, keepdims=True, eps=eps)


class HypersphericalDense(nn.Module):
    """Dense layer with hyperspherical weight normalization.

    For a kernel of shape (in_features, out_features), we normalize each output
    column vector to unit norm and then scale by a learnable per-output kappa.
    This constrains weight norm growth while allowing controlled scaling.
    """

    features: int
    use_bias: bool = False
    eps: float = 1e-6
    kernel_init: Callable = nn.initializers.orthogonal()
    bias_init: Callable = nn.initializers.zeros
    dtype: jax.typing.DTypeLike | None = None
    param_dtype: jax.typing.DTypeLike = jnp.float32
    precision: jax.lax.PrecisionLike = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        in_features = inputs.shape[-1]
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (in_features, self.features),
            self.param_dtype,
        )
        # Normalize each output column: norm over input dimension (axis=0).
        kernel_unit = kernel / _safe_norm(kernel, axis=0, keepdims=True, eps=self.eps)

        kappa = self.param(
            "kappa",
            nn.initializers.ones,
            (self.features,),
            self.param_dtype,
        )
        kernel_h = kernel_unit * kappa[None, :]

        y = jax.lax.dot_general(
            inputs,
            kernel_h,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
            bias = jnp.asarray(bias, dtype=inputs.dtype)
            y = y + jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class Shift(nn.Module):
    c_shift: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype) * self.c_shift
        return jnp.concatenate([inputs, new_axis], axis=-1)


class Scaler(nn.Module):
    dim: int
    init: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.param("scale", nn.initializers.ones, (self.dim,)) * self.init * inputs


class LERP(nn.Module):
    features: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, start: jnp.ndarray, end: jnp.ndarray) -> jnp.ndarray:
        return start + Scaler(self.features, init=0.5)(end - start)


class SimbaV2Embedding(nn.Module):
    hidden_dim: int
    c_shift: float = 3.0
    kernel_init: Callable = nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = Shift(self.c_shift)(inputs)
        x = l2_normalize(x, axis=-1)
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)
        return l2_normalize(Scaler(self.hidden_dim)(x), axis=-1)


class SimbaV2Block(nn.Module):
    hidden_dim: int
    hidden_multiplier: int = 4
    kernel_init: Callable = nn.initializers.orthogonal()
    eps: float = 1e-6
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = HypersphericalDense(
            self.hidden_dim * self.hidden_multiplier,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(inputs)
        x = Scaler(self.hidden_dim * self.hidden_multiplier)(x)
        x = self.activation(x)
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(x)
        x = l2_normalize(x, axis=-1)
        return l2_normalize(LERP(self.hidden_dim, eps=self.eps)(inputs, x), axis=-1)


class SimbaV2Head(nn.Module):
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = nn.initializers.orthogonal()
    use_bias: bool = False
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = HypersphericalDense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
        )(inputs)
        return HypersphericalDense(
            self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(Scaler(self.hidden_dim)(x))


def network_body(
    features: jnp.ndarray, network: ModelConfig, layer: type[nn.Module] = Dense
) -> jnp.ndarray:
    """Build a configured body inside the caller's compact scope."""
    if isinstance(network, MLPConfig):
        for hidden in network.layers:
            features = ACTIVATIONS[hidden.activation](layer(hidden.units)(features))
        return features
    if network.kind == "simba":
        features = layer(network.blocks[0])(features)
        for width in network.blocks:
            if features.shape[-1] != width:
                features = layer(width)(features)
            features = ResidualBlock(width, activation=ACTIVATIONS[network.activation])(features)
        return nn.LayerNorm()(features)
    if network.kind == "simbav2":
        features = SimbaV2Embedding(network.blocks[0])(features)
        for width in network.blocks:
            if features.shape[-1] != width:
                features = l2_normalize(HypersphericalDense(width, use_bias=False)(features))
            features = SimbaV2Block(width, activation=ACTIVATIONS[network.activation])(features)
        return features
    raise ValueError(f"Unsupported network body: {network.kind!r}")
