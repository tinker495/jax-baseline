import jax
import jax.numpy as jnp


def clip_uniform(min_val=-1.0, max_val=1.0):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, min_val, max_val)

    return init


def clip_factorized_uniform(magnitude: float = 1.0):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] == 0:
            raise ValueError("Cannot initialize factorized uniform with zero-sized input dimension")
        limit = magnitude / jnp.sqrt(shape[0])
        return jax.random.uniform(key, shape, dtype, -limit, limit)

    return init
