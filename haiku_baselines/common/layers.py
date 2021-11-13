import numpy as np
import abc as ABC
import haiku as hk
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Any, Callable, Iterable, Optional, Type

def get_eps(n):
    key = hk.next_rng_key()
    x = jax.random.normal(key,n,dtype=jax.float32)
    return jnp.sign(x)*jnp.sqrt(jnp.abs(x))

class NoisyLinear(hk.Module):
    """Noisy Linear module"""
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        ):
        super(NoisyLinear, self).__init__()
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        
    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
        ) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w_mu = hk.get_parameter("w_mu", [input_size, output_size], dtype, init=w_init)
        w_sigma = hk.get_parameter("w_sigma", [input_size, output_size], dtype, init=w_init)
        
        eps_in = get_eps(input_size)
        eps_out = get_eps(output_size)
        eps_ij = jnp.outer(eps_in,eps_out)
        out = jnp.dot(inputs, w_mu + w_sigma*eps_ij, precision=precision)

        if self.with_bias:
            b_mu = hk.get_parameter("b_mu", [self.output_size], dtype, init=self.b_init)
            b_sigma = hk.get_parameter("b_sigma", [self.output_size], dtype, init=self.b_init)
            b_mu = jnp.broadcast_to(b_mu, out.shape)
            b_sigma = jnp.broadcast_to(b_sigma, out.shape)
            out = out + b_mu + b_sigma*eps_out
        return out