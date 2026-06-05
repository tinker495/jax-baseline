import jax
import jax.numpy as jnp


def truncated_mixture(quantiles, cut):
    """Concatenates and sorts quantile values, then truncates the highest values.

    Used in TQC and CrossQ_TQC algorithms to implement truncated quantile critics.

    Args:
        quantiles: List of quantile values from multiple critics to be mixed
        cut: Number of highest quantile values to remove

    Returns:
        Sorted and truncated quantile values with the highest 'cut' values removed
    """
    quantiles = jnp.concatenate(quantiles, axis=1)
    sorted = jnp.sort(quantiles, axis=1)
    return sorted[:, :-cut]


def q_log_pi(q, entropy_tau):
    q_submax = q - jnp.max(q, axis=1, keepdims=True)
    logsum = jax.nn.logsumexp(q_submax / entropy_tau, axis=1, keepdims=True)
    tau_log_pi = q_submax - entropy_tau * logsum
    return q_submax, tau_log_pi


def kl_divergence_discrete(p, q, eps: float = 1e-8):
    # Add epsilon to prevent log(0)
    p_safe = p + eps
    q_safe = q + eps
    # Compute log values
    log_p = jnp.log(p_safe)
    log_q = jnp.log(q_safe)
    # Compute KL divergence with masking
    return jnp.sum(p * (log_p - log_q))


def kl_divergence_continuous(p, q):
    p_mu, p_std = p
    q_mu, q_std = q
    term1 = jnp.log(q_std / p_std)
    term2 = (p_std**2 + (p_mu - q_mu) ** 2) / (2 * q_std**2)
    return jnp.sum(term1 + term2 - 0.5, axis=-1, keepdims=True)
