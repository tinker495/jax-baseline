import jax
import jax.numpy as jnp


def clip_factorized_uniform(magnitude: float = 1):
    def init(key, shape, dtype=jnp.float32):
        min_val = -magnitude / jnp.sqrt(shape[0])
        max_val = magnitude / jnp.sqrt(shape[0])
        return jax.random.uniform(key, shape, dtype, min_val, max_val)

    return init
