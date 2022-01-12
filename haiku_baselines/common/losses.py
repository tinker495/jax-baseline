from typing import Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def HuberLosses(error,quantile,delta):
    huber = ((jnp.abs(error) <= delta).astype(jnp.float32) *
            0.5 * error ** 2 +
            (jnp.abs(error) > delta).astype(jnp.float32) *
            delta * (jnp.abs(error) - 0.5 * delta))
    mul = jax.lax.stop_gradient(jnp.abs(quantile - (error < 0).astype(jnp.float32)))
    return jnp.sum(jnp.mean(mul*huber,axis=1),axis=1)