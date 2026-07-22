import jax
import jax.numpy as jnp
import numpy as np


def entropy_target_from_sigma(action_dim: int, sigma_target: float) -> float:
    if sigma_target <= 0:
        raise ValueError("sigma_target must be greater than 0")
    return 0.5 * action_dim * np.log(2.0 * np.pi * np.e * sigma_target**2)


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
    sorted_quantiles = jnp.sort(quantiles, axis=1)
    return sorted_quantiles[:, :-cut]


def q_log_pi(q, entropy_tau, clip=False, min_clip=-1.0, max_clip=0.0):
    q_submax = q - jnp.max(q, axis=1, keepdims=True)
    logsum = jax.nn.logsumexp(q_submax / entropy_tau, axis=1, keepdims=True)
    tau_log_pi = q_submax - entropy_tau * logsum
    # Munchausen RL (arXiv:2007.14430, Eq. 3): the scaled log-policy is unclipped by
    # default (soft-bootstrap term); the reward log-policy term passes clip=True to bound it.
    if clip:
        tau_log_pi = jnp.clip(tau_log_pi, min_clip, max_clip)
    return q_submax, tau_log_pi


def kl_divergence_discrete(p, q, eps: float = 1e-8):
    # eps guards log(0); the outer weight uses the raw p so zero-mass atoms drop out.
    log_p = jnp.log(p + eps)
    log_q = jnp.log(q + eps)
    return jnp.sum(p * (log_p - log_q))


def kl_divergence_continuous(p, q):
    p_mu, p_std = p
    q_mu, q_std = q
    term1 = jnp.log(q_std / p_std)
    term2 = (p_std**2 + (p_mu - q_mu) ** 2) / (2 * q_std**2)
    return jnp.sum(term1 + term2 - 0.5, axis=-1, keepdims=True)
