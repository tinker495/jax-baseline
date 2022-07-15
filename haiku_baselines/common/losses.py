from typing import Generator, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def hubberloss(x, delta):
    abs_errors = jnp.abs(x)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic ** 2 + delta * linear

def log_cosh(x):
    return jnp.logaddexp(x, -x) - jnp.log(2.0).astype(x.dtype)

def QuantileHuberLosses(q_tile, target_tile,quantile,delta):
    error = target_tile - q_tile
    error_neg = (error < 0.).astype(jnp.float32)
    weight = jnp.abs(quantile - error_neg)
    huber = hubberloss(error,delta) / delta
    return jnp.sum(jnp.mean(weight*huber,axis=1),axis=1)

def QuantileSquareLosses(q_tile, target_tile,quantile,delta):
    error = target_tile - q_tile
    error_neg = (error < 0.).astype(jnp.float32)
    weight = jnp.abs(quantile - error_neg)
    square = jnp.square(error)
    return jnp.sum(jnp.mean(weight*square,axis=1),axis=1)

def FQFQuantileLosses(tau_vals, vals, quantile):
    values_1 = tau_vals - vals[:,:-1]
    sign_1 = (tau_vals > jnp.concatenate([vals[:, :1], tau_vals[:, :-1]], axis=1))
    
    values_2 = tau_vals - vals[:,1:]
    sign_2 = (tau_vals < jnp.concatenate([tau_vals[:, 1:], vals[:, -1:]], axis=1))
    
    grad_of_taus = jnp.where(sign_1, values_1, -values_1) + jnp.where(sign_2, values_2, -values_2)
    return jnp.sum(grad_of_taus * quantile[:,1:-1],axis=1)