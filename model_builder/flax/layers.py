import flax.linen as nn
import jax
import jax.numpy as jnp
from model_builder.flax.initializers import clip_factorized_uniform
from typing import Callable
from flax.linen.dtypes import promote_dtype
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
            jax.nn.initializers.constant(SIGMA_INIT/jnp.sqrt(input_size)),
            (input_size, self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias_mu = self.param("bias_mu", self.bias_init, (self.features,), self.param_dtype)
            bias_sigma = self.param(
                "bias_sigma", jax.nn.initializers.constant(SIGMA_INIT/jnp.sqrt(input_size)), (self.features,), self.param_dtype
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
