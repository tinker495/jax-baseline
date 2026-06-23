from __future__ import annotations

import optax

from jax_baselines.optim import OptimizerFactory, adopt, optimizer_reset_by_period


def _require_contrib_transform(name: str):
    transform = getattr(optax.contrib, name, None)
    if transform is None:
        raise ValueError(
            f"Optimizer '{name}' requires optax.contrib.{name}, "
            "which is unavailable in the installed optax version."
        )
    return transform


def select_optimizer(optim_str, lr, eps=1e-2 / 256.0, weight_decay=1e-4, grad_max=None):
    """
    Select an optimizer based on experiment adapter policy.

    ``optim_str`` may include a ``_reset_<steps>`` suffix, e.g. ``adam_reset_2000``.
    Optimizer names, default epsilon/weight decay, clipping, and reset parsing belong
    to experiments; algorithm core receives only the resulting transformation/factory.
    """

    optim = None
    reset_steps = None
    if "_reset_" in optim_str:
        optim_str, reset_steps = optim_str.split("_reset_")
        reset_steps = int(reset_steps)

    match optim_str:
        case "adam":
            optim = optax.adam(lr, b1=0.9, b2=0.999, eps=eps)
        case "nadam":
            optim = optax.adam(lr, b1=0.9, b2=0.999, eps=eps, nesterov=True)
        case "schedule_free_adam":
            optim = optax.contrib.schedule_free_adamw(lr, b1=0.9, b2=0.999, eps=eps)
        case "adopt":
            optim = adopt(lr, b1=0.9, b2=0.9999, eps=eps)
        case "nadopt":
            optim = adopt(lr, b1=0.9, b2=0.9999, eps=eps, nesterov=True)
        case "adamw":
            optim = optax.adamw(lr, b1=0.9, b2=0.999, eps=eps, weight_decay=weight_decay)
        case "ano":
            optim = _require_contrib_transform("ano")(lr, weight_decay=weight_decay)
        case "rmsprop":
            optim = optax.rmsprop(lr, eps=eps)
        case "sgd":
            optim = optax.sgd(lr)
        case "adabelief":
            optim = optax.adabelief(lr, eps=eps)
        case "lion":
            optim = optax.lion(lr, weight_decay=1e-5)
        case "prodigy":
            optim = optax.contrib.prodigy(lr, eps=eps, weight_decay=1e-4)
        case "muon":
            optim = optax.contrib.muon(lr, weight_decay=weight_decay)
        case "normuon":
            optim = _require_contrib_transform("normuon")(lr, weight_decay=weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: {optim_str}")

    if grad_max is not None:
        optim = optax.chain(optax.clip_by_global_norm(grad_max), optim)

    if reset_steps is not None:
        optim = optimizer_reset_by_period(optim, reset_steps)

    return optim


def make_optimizer_factory(
    optim_str: str,
    *,
    eps: float = 1e-2 / 256.0,
    weight_decay: float = 1e-4,
    grad_max: float | None = None,
) -> OptimizerFactory:
    def factory(lr):
        return select_optimizer(
            optim_str,
            lr,
            eps=eps,
            weight_decay=weight_decay,
            grad_max=grad_max,
        )

    return factory


def make_batch_scaled_optimizer_factory(
    optim_str: str,
    batch_size: int,
    *,
    weight_decay: float = 1e-4,
    grad_max: float | None = None,
) -> OptimizerFactory:
    return make_optimizer_factory(
        optim_str,
        eps=1e-3 / batch_size,
        weight_decay=weight_decay,
        grad_max=grad_max,
    )


__all__ = [
    "make_batch_scaled_optimizer_factory",
    "make_optimizer_factory",
    "select_optimizer",
]
