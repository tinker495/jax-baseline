import jax
import jax.numpy as jnp


def clip_uniform_initializers(min_val=-1, max_val=1):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, min_val, max_val)

    return init

def clip_factorized_uniform():
    
    def init(key, shape, dtype=jnp.float32):
        min_val = -1/jnp.sqrt(shape[0])
        max_val = 1/jnp.sqrt(shape[0])
        return jax.random.uniform(key, shape, dtype, min_val, max_val)

    return init