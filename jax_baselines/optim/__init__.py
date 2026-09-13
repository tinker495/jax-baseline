"""Core optimizer numerics and optimizer-factory protocol.

Algorithm families intentionally receive optimizer factories from experiment
adapters. String names, defaults, clipping policy, and reset-suffix parsing are
experiment policy and must not be resolved in core constructors.
"""

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import optax


class OptimizerFactory(Protocol):
    def __call__(self, learning_rate: optax.ScalarOrSchedule) -> optax.GradientTransformation:
        pass


def require_optimizer_factory(
    optimizer_factory: OptimizerFactory | None,
) -> OptimizerFactory:
    """Return a supplied optimizer factory or fail with the migration contract."""

    if optimizer_factory is None:
        raise ValueError(
            "optimizer_factory is required; resolve optimizer names/defaults in experiments "
            "and pass the resulting factory into the algorithm family."
        )
    return optimizer_factory


class OptimizerMetricsState(NamedTuple):
    inner_state: optax.OptState
    count: jax.Array
    grad_norm_pre_clip: jax.Array
    update_norm: jax.Array
    parameter_norm: jax.Array
    learning_rate: jax.Array
    grad_clip_fraction: jax.Array


def track_optimizer(
    optimizer: optax.GradientTransformation,
    learning_rate: optax.ScalarOrSchedule,
    grad_max: float | None = None,
    reset_steps: int | None = None,
) -> optax.GradientTransformation:
    """Observe optimizer inputs and updates without changing their values.

    Learning rate is the supplied scalar schedule, before any adaptive
    preconditioning. Parameter norm is measured before the update. A periodic
    inner-state reset also restarts the schedule's diagnostic counter.
    """
    if reset_steps is not None and reset_steps <= 0:
        raise ValueError("optimizer reset_steps must be positive")

    def rate(count):
        if reset_steps is not None:
            count = count % reset_steps
        return jnp.asarray(learning_rate(count) if callable(learning_rate) else learning_rate)

    def init_fn(params):
        zero = jnp.asarray(0.0)
        return OptimizerMetricsState(
            optimizer.init(params),
            jnp.zeros((), dtype=jnp.int32),
            zero,
            zero,
            optax.tree.norm(params),
            rate(jnp.zeros((), dtype=jnp.int32)),
            zero,
        )

    def update_fn(updates, state, params=None):
        grad_norm = optax.tree.norm(updates)
        updates, inner_state = optimizer.update(updates, state.inner_state, params)
        return updates, OptimizerMetricsState(
            inner_state,
            jnp.asarray(optax.safe_increment(state.count)),
            grad_norm,
            optax.tree.norm(updates),
            optax.tree.norm(params) if params is not None else jnp.asarray(jnp.nan),
            rate(state.count),
            (grad_norm > grad_max).astype(jnp.float32)
            if grad_max is not None
            else jnp.asarray(0.0),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def optimizer_metrics(opt_state: optax.OptState, prefix: str) -> dict[str, jax.Array]:
    """Read the last update; arbitrary injected Optax optimizers may opt out."""
    if isinstance(opt_state, optax.InjectStatefulHyperparamsState):
        opt_state = opt_state.inner_state
    if not isinstance(opt_state, OptimizerMetricsState):
        return {}
    return {
        f"optim/{prefix}_grad_norm_pre_clip": opt_state.grad_norm_pre_clip,
        f"optim/{prefix}_update_norm": opt_state.update_norm,
        f"optim/{prefix}_parameter_norm": opt_state.parameter_norm,
        f"optim/{prefix}_learning_rate": opt_state.learning_rate,
        f"optim/{prefix}_grad_clip_fraction": opt_state.grad_clip_fraction,
    }


def optimizer_reset_by_period(
    optimizer: optax.GradientTransformation, reset_steps: int
) -> optax.GradientTransformation:
    """Create an optimizer wrapper that periodically resets optimizer state."""

    def init_fn(params):
        opt_state = optimizer.init(params)
        return (opt_state, jnp.zeros((), dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("periodic optimizer reset requires parameters")
        opt_state, step_count = state
        updates, opt_state = optimizer.update(updates, opt_state, params)

        def reset():
            return optimizer.init(params)

        def keep():
            return opt_state

        opt_state = jax.lax.cond((step_count + 1) % reset_steps == 0, reset, keep)

        return updates, (opt_state, step_count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = [
    "OptimizerFactory",
    "OptimizerMetricsState",
    "optimizer_metrics",
    "optimizer_reset_by_period",
    "require_optimizer_factory",
    "track_optimizer",
]
