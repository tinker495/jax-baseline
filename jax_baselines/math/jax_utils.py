import jax.numpy as jnp


def convert_jax(obs: list):
    return [
        o.astype(jnp.float32) if o.dtype != jnp.uint8 else o.astype(jnp.float32) / 255.0
        for o in obs
    ]
