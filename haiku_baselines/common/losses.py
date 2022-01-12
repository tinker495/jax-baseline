from typing import Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def QuantileHuberLosses(q_tile, target_tile,quantile,delta):
    error = target_tile - q_tile
    error_neg = jax.lax.stop_gradient((error < 0.).astype(jnp.float32))
    weight = jnp.abs(quantile - error_neg)
    huber = ((jnp.abs(error) <= delta).astype(jnp.float32) *
            0.5 * error ** 2 +
            (jnp.abs(error) > delta).astype(jnp.float32) *
            delta * (jnp.abs(error) - 0.5 * delta))
    return jnp.sum(jnp.mean(weight*huber,axis=1),axis=1)