import jax.numpy as jnp


def convert_jax(obs: list):
    return [o.astype(jnp.float32) for o in obs]
