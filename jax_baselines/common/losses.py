import jax.numpy as jnp


def hubberloss(x, delta):
    abs_errors = jnp.abs(x)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def log_cosh(x):
    return jnp.logaddexp(x, -x) - jnp.log(2.0).astype(x.dtype)


def QuantileHuberLosses(target_tile, q_tile, quantile, delta):
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
    return jnp.sum(jnp.mean(weight * huber, axis=1), axis=1)


def FQFQuantileLosses(tau_vals, tau_hat_val, quantile):
    values_1 = tau_vals - tau_hat_val[:, :-1]
    sign_1 = tau_vals > jnp.concatenate([tau_hat_val[:, :1], tau_vals[:, :-1]], axis=1)

    values_2 = tau_vals - tau_hat_val[:, 1:]
    sign_2 = tau_vals < jnp.concatenate([tau_vals[:, 1:], tau_hat_val[:, -1:]], axis=1)

    grad_of_taus = jnp.where(sign_1, values_1, -values_1) + jnp.where(sign_2, values_2, -values_2)
    return jnp.sum(grad_of_taus * quantile[:, 1:-1], axis=1)
