"""Scalar learner diagnostics computed from the tensors used by an update."""

import jax
import jax.numpy as jnp


def array_metrics(values: jax.Array, prefix: str) -> dict[str, jax.Array]:
    return {
        f"{prefix}_mean": jnp.mean(values),
        f"{prefix}_std": jnp.std(values),
        f"{prefix}_min": jnp.min(values),
        f"{prefix}_max": jnp.max(values),
    }


def td_metrics(
    predictions: jax.Array, targets: jax.Array, prefix: str = "loss/td"
) -> dict[str, jax.Array]:
    """Summarize target minus prediction, independently of the optimized loss."""
    residual = jnp.ravel(targets) - jnp.ravel(predictions)
    return {
        f"{prefix}_signed_mean": jnp.mean(residual),
        f"{prefix}_abs_mean": jnp.mean(jnp.abs(residual)),
        f"{prefix}_abs_p95": jnp.percentile(jnp.abs(residual), 95),
    }


def rollout_metrics(values, targets, raw_advantages) -> dict[str, jax.Array]:
    target_variance = jnp.var(targets)
    return {
        "loss/explained_variance": jnp.where(
            target_variance > 0,
            1 - jnp.var(jnp.ravel(targets) - jnp.ravel(values)) / target_variance,
            jnp.nan,
        ),
        "loss/value_mean": jnp.mean(values),
        "loss/value_std": jnp.std(values),
        "loss/mean_target": jnp.mean(targets),
        "loss/target_std": jnp.std(targets),
        "loss/advantage_mean": jnp.mean(raw_advantages),
        "loss/advantage_std": jnp.std(raw_advantages),
    }


def gaussian_metrics(log_std: jax.Array) -> dict[str, jax.Array]:
    std = jnp.exp(log_std)
    return {
        "loss/policy_std_mean": jnp.mean(std),
        "loss/policy_std_min": jnp.min(std),
        "loss/policy_std_max": jnp.max(std),
        "loss/log_std_mean": jnp.mean(log_std),
        "loss/log_std_min": jnp.min(log_std),
        "loss/log_std_max": jnp.max(log_std),
    }


def policy_ratio_metrics(new_log_prob, old_log_prob, clip_eps) -> dict[str, jax.Array]:
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    return {
        "loss/approx_kl": jnp.mean(jnp.expm1(log_ratio) - log_ratio),
        "loss/clip_fraction": jnp.mean(jnp.abs(ratio - 1) > clip_eps),
    }


def replay_metrics(weights, priorities) -> dict[str, jax.Array]:
    """Priorities submitted by the learner; ESS concerns sampled IS weights only."""
    priorities = jnp.ravel(priorities)
    weights = jnp.broadcast_to(jnp.ravel(weights), priorities.shape)
    return {
        "loss/priority_mean": jnp.mean(priorities),
        "loss/priority_max": jnp.max(priorities),
        "loss/priority_p95": jnp.percentile(priorities, 95),
        "loss/is_weight_mean": jnp.mean(weights),
        "loss/is_weight_min": jnp.min(weights),
        "loss/is_weight_max": jnp.max(weights),
        "loss/is_weight_ess": jnp.square(jnp.sum(weights)) / jnp.sum(jnp.square(weights)),
    }


def categorical_metrics(
    probabilities, target_probs, support, prefix: str = "loss"
) -> dict[str, jax.Array]:
    if probabilities.shape[-1] != support.size:
        raise ValueError("categorical support must have one value per probability bin")
    target_log = jnp.log(jnp.maximum(target_probs, jnp.finfo(target_probs.dtype).tiny))
    online_log = jnp.log(jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny))
    return {
        f"{prefix}/categorical_cross_entropy": -jnp.mean(
            jnp.sum(target_probs * online_log, axis=-1)
        ),
        f"{prefix}/categorical_kl": jnp.mean(
            jnp.sum(target_probs * (target_log - online_log), axis=-1)
        ),
        f"{prefix}/target_entropy": -jnp.mean(jnp.sum(target_probs * target_log, axis=-1)),
        f"{prefix}/online_edge_mass_low": jnp.mean(probabilities[..., 0]),
        f"{prefix}/online_edge_mass_high": jnp.mean(probabilities[..., -1]),
        f"{prefix}/target_edge_mass_low": jnp.mean(target_probs[..., 0]),
        f"{prefix}/target_edge_mass_high": jnp.mean(target_probs[..., -1]),
    }


def support_metrics(
    probabilities, target_atoms, support_min, support_max, prefix: str = "loss"
) -> dict[str, jax.Array]:
    """Probability mass outside the support before projection, summed per sample."""
    return {
        f"{prefix}/support_clipped_mass_low": jnp.mean(
            jnp.sum(
                (probabilities * (target_atoms < support_min)).reshape(probabilities.shape[0], -1),
                axis=1,
            )
        ),
        f"{prefix}/support_clipped_mass_high": jnp.mean(
            jnp.sum(
                (probabilities * (target_atoms > support_max)).reshape(probabilities.shape[0], -1),
                axis=1,
            )
        ),
    }


def quantile_metrics(quantiles, taus=None, prefix: str = "loss") -> dict[str, jax.Array]:
    """Distribution width and adjacent crossings, with outputs ordered by tau."""
    if taus is not None:
        quantiles = jnp.take_along_axis(
            quantiles, jnp.argsort(jnp.broadcast_to(taus, quantiles.shape), axis=-1), axis=-1
        )
    return {
        f"{prefix}/quantile_spread": jnp.mean(
            jnp.max(quantiles, axis=-1) - jnp.min(quantiles, axis=-1)
        ),
        f"{prefix}/quantile_crossing_rate": (
            jnp.mean(quantiles[..., :-1] > quantiles[..., 1:])
            if quantiles.shape[-1] > 1
            else jnp.asarray(0.0)
        ),
    }
