from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu


def adopt(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = False,
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
    mu_dtype: Optional[jnp.dtype] = None,
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

    Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.
    use_clipping: Whether to use gradient clipping to improve stability.
        When enabled, the clipping value is determined by the clip_value_fn.
    clip_value_fn: A function that takes a step index and returns a clipping value.
        Default is :math:`x^{0.25}`

    Returns:
    A :class:`optax.GradientTransformation` object.
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
        if nesterov:
            mu_ = otu.tree_update_moment(mu_updates, mu, b1_, 1)
        else:
            mu_ = mu
        updates = mu_
        mu = otu.tree_cast(mu, mu_dtype)
        return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def optimizer_reset_by_period(
    optimizer: optax.GradientTransformation, reset_steps: int
) -> optax.GradientTransformation:
    """Creates an optimizer that periodically resets its parameters.

    Args:
        optimizer: Base optimizer to wrap
        reset_steps: Number of steps between parameter resets

    Returns:
        An optax.GradientTransformation that wraps the base optimizer with periodic resets
    """

    def init_fn(params):
        opt_state = optimizer.init(params)
        return (opt_state, jnp.zeros((), dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        opt_state, step_count = state
        updates, opt_state = optimizer.update(updates, opt_state, params)

        def reset():
            return optimizer.init(params)

        def keep():
            return opt_state

        opt_state = jax.lax.cond((step_count + 1) % reset_steps == 0, reset, keep)

        return updates, (opt_state, step_count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def select_optimizer(optim_str, lr, eps=1e-2 / 256.0, grad_max=None):
    """
    Selects an optimizer based on the optimizer string.
    If reset_steps is not None, it wraps the optimizer with periodic parameter resets.

    Args:
        optim_str: Name of the optimizer input_example "adam", "adam_reset_2000"
            "_" is used to separate the optimizer name and the reset steps
        lr: Learning rate
        eps: Epsilon for Adam and Prodigy
        grad_max: Gradient clipping value
    """

    lr_schedule = optax.linear_schedule(init_value=0, end_value=lr, transition_steps=1000)
    # warmup

    optim = None
    reset_steps = None
    if "_reset_" in optim_str:
        optim_str, reset_steps = optim_str.split("_reset_")
        reset_steps = int(reset_steps)

    match optim_str:
        case "adam":
            optim = optax.adam(lr_schedule, b1=0.9, b2=0.999, eps=eps)
        case "adam_low_b1":
            optim = optax.adam(lr_schedule, b1=0.5, b2=0.999, eps=eps)
        case "adopt":
            optim = adopt(lr_schedule, b1=0.9, b2=0.9999, eps=eps)
        case "adamw":
            optim = optax.adamw(lr_schedule, b1=0.9, b2=0.999, eps=eps, weight_decay=1e-4)
        case "rmsprop":
            optim = optax.rmsprop(lr_schedule, eps=eps)
        case "sgd":
            optim = optax.sgd(lr_schedule)
        case "adabelief":
            optim = optax.adabelief(lr_schedule, eps=eps)
        case "lion":
            optim = optax.lion(lr_schedule, weight_decay=1e-5)
        case "prodigy":
            lr_schedule = optax.linear_schedule(init_value=0, end_value=0.5, transition_steps=1000)
            optim = optax.contrib.prodigy(lr_schedule, eps=eps, weight_decay=1e-4)
        case _:
            raise ValueError(f"Unknown optimizer: {optim_str}")

    if grad_max is not None:
        optim = optax.chain(optax.clip_by_global_norm(grad_max), optim)

    if reset_steps is not None:
        optim = optimizer_reset_by_period(optim, reset_steps)

    return optim
