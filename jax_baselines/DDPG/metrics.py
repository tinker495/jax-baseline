"""DPG diagnostics from the samples used by each optimizer update."""

import jax.numpy as jnp

from jax_baselines.math.metrics import (
    array_metrics,
    gaussian_metrics,
    replay_metrics,
    td_metrics,
)


def critic_metrics(q_values, targets, losses, weights, priorities=None, loss_scale=1.0):
    targets = jnp.asarray(targets).reshape(-1)
    values = [jnp.asarray(q).reshape(-1) for q in q_values]
    metrics = {
        "loss/target_std": jnp.std(targets),
        "loss/target_min": jnp.min(targets),
        "loss/target_max": jnp.max(targets),
        "loss/unweighted_loss": loss_scale * sum(jnp.mean(loss) for loss in losses),
        **td_metrics(jnp.stack(values), jnp.broadcast_to(targets, (len(values), targets.size))),
    }
    for index, (q, loss) in enumerate(zip(values, losses, strict=True), start=1):
        metrics.update(array_metrics(q, f"loss/q{index}"))
        metrics.update(td_metrics(q, targets, f"loss/td{index}"))
        metrics[f"loss/critic{index}_loss"] = jnp.mean(jnp.asarray(weights).squeeze() * loss)
    if len(values) == 2:
        metrics["loss/q_disagreement"] = jnp.mean(jnp.abs(values[0] - values[1]))
    if priorities is not None:
        metrics.update(replay_metrics(weights, priorities))
    return metrics


def stochastic_actor_metrics(log_prob, log_std, q_value, ent_coef, target_entropy):
    return {
        "loss/entropy": -jnp.mean(log_prob),
        "loss/entropy_gap": -jnp.mean(log_prob) - target_entropy,
        "loss/policy_target_entropy": jnp.asarray(target_entropy),
        "loss/actor_q_mean": jnp.mean(q_value),
        "loss/actor_entropy_term": jnp.mean(ent_coef * log_prob),
        **gaussian_metrics(log_std),
    }


def reduce_metrics(metrics, metric_counts):
    """Average observations; a skipped optimizer update contributes no observation."""
    counts = {name: jnp.sum(metric_counts[name]) for name in metrics}
    return {
        name: jnp.sum(jnp.where(metric_counts[name] > 0, value * metric_counts[name], 0))
        / jnp.maximum(counts[name], 1)
        for name, value in metrics.items()
    }, counts
