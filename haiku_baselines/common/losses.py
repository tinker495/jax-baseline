from typing import Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def hubberloss(x, delta):
  abs_x = jnp.abs(x)
  quadratic = jnp.minimum(abs_x, delta)
  # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
  linear = abs_x - quadratic
  return 0.5 * quadratic**2 + delta * linear

def QuantileHuberLosses(q_tile, target_tile,quantile,delta):
    error = target_tile - q_tile
    error_neg = jax.lax.stop_gradient((error < 0.).astype(jnp.float32))
    weight = jnp.abs(quantile - error_neg)
    huber = hubberloss(error,delta)
    return jnp.sum(jnp.mean(weight*huber,axis=1),axis=1)