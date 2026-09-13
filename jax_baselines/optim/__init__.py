"""Core optimizer numerics and optimizer-factory protocol.

Algorithm families intentionally receive optimizer factories from experiment
adapters. String names, defaults, clipping policy, and reset-suffix parsing are
experiment policy and must not be resolved in core constructors.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu


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


def adopt(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: Any | None = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
) -> optax.GradientTransformationExtraArgs:
    return optax.chain(
        scale_by_adopt(
            b1=b1,
            b2=b2,
            eps=eps,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            use_clipping=use_clipping,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )


def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: jnp.dtype | None = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x**0.25,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the ADOPT algorithm.

    ADOPT (Modified Adam Can Converge with Any beta2 with the Optimal Rate) is a variant
    of Adam that can converge with any beta2 value while maintaining the optimal rate.

    This implementation includes a clipping operation to improve stability, especially
    in the early stages of training. The clipping helps avoid near-zero divisions when
    some elements of the parameter gradient are near zero at initialization.
    """

    mu_dtype = optax._src.utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        b2_ = jnp.where(state.count > 0, b2, 0)
        b1_ = jnp.where(state.count > 0, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2_, 2)
        if use_clipping:
            clip_value = clip_value_fn(state.count)
            mu_updates = jax.tree.map(
                lambda ud, nu: jnp.clip(
                    ud / jnp.maximum(jnp.sqrt(nu), eps), -clip_value, clip_value
                ),
                updates,
                state.nu,
            )
        else:
            mu_updates = jax.tree.map(
                lambda ud, nu: ud / jnp.maximum(jnp.sqrt(nu), eps), updates, state.nu
            )
        mu = otu.tree_update_moment(mu_updates, state.mu, b1_, 1)
        count_inc = optax._src.numerics.safe_increment(state.count)
        mu_ = otu.tree_update_moment(mu_updates, mu, b1_, 1) if nesterov else mu
        updates = mu_
        mu = otu.tree_cast(mu, mu_dtype)
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


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
    "adopt",
    "optimizer_metrics",
    "optimizer_reset_by_period",
    "require_optimizer_factory",
    "scale_by_adopt",
    "track_optimizer",
]
