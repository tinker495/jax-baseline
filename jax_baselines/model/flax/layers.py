import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype


class NoisyDense(nn.Dense):
    rng_collection: str = "noisy"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

    def get_eps(self, n):
        key = nn.Module.make_rng(self.rng_collection)
        x = jax.random.normal(key, (n,), dtype=jnp.float32)
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))
