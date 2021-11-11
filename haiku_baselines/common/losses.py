from typing import Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def MSELosses(input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(input - target),axis=1)

@jax.jit
def HuberLosses(input: jnp.ndarray, target: jnp.ndarray, beta: float = 1) -> jnp.ndarray:
    error = input - target
    return jnp.mean(jnp.where(jnp.abs(error) <= beta, 0.5*jnp.square(error), beta*(jnp.abs(error) - 0.5*beta)),axis=1)