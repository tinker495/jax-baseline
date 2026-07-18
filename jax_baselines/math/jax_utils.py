import jax.numpy as jnp


def convert_jax(obs: dict):
    return {
        key: value.astype(jnp.float32)
        if value.dtype != jnp.uint8
        else value.astype(jnp.float32) / 255.0
        for key, value in obs.items()
    }
