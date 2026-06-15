"""Compatibility exports for optimizer numerics.

Optimizer string-selection policy moved to :mod:`experiments.optimizers`.
Algorithm core must receive optimizer factories instead of importing a selector
from this common module.
"""

from jax_baselines.optim import adopt, optimizer_reset_by_period, scale_by_adopt

__all__ = ["adopt", "optimizer_reset_by_period", "scale_by_adopt"]
