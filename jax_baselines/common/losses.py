import chex
import jax
import jax.numpy as jnp


def hubberloss(x, delta):
    abs_errors = jnp.abs(x)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def log_cosh(x):
    return jnp.logaddexp(x, -x) - jnp.log(2.0).astype(x.dtype)


def QuantileHuberLosses(target_tile, q_tile, quantile, delta, target_tile_weight=None):
    """Compute the quantile huber loss for quantile regression.

    Args:
        target_tile: target tile (batch_size, num_tau_prime, 1)
        q_tile: quantile tile (batch_size, 1, num_tau)
        quantile: quantile (batch_size, 1, num_tau)
        delta: huber loss delta (float)

    Returns:
        quantile huber loss (batch_size)
    """
    error = target_tile - q_tile
    error_neg = (error < 0.0).astype(jnp.float32)
    weight = jnp.abs(quantile - error_neg)
    huber = hubberloss(error, delta) / delta
    if target_tile_weight is None:
        return jnp.sum(jnp.mean(weight * huber, axis=1), axis=1)
    else:
        target_tile_weight = target_tile_weight / jnp.mean(
            target_tile_weight, axis=1, keepdims=True
        )
        return jnp.sum(jnp.mean(weight * huber * target_tile_weight, axis=1), axis=1)


def FQFQuantileLosses(tau_vals, tau_hat_vals, tau):
    """Compute the fqf loss."""
    grad_of_taus = jax.lax.stop_gradient(2 * tau_vals - tau_hat_vals[:, :-1] - tau_hat_vals[:, 1:])
    chex.assert_shape(grad_of_taus, tau[:, 1:-1].shape)
    return jnp.sum(grad_of_taus * tau[:, 1:-1], axis=1)
