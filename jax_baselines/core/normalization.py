"""Observation and reward normalization state shared by algorithm families.

Statistics always live on the compute device as float32 JAX arrays, whatever the replay
storage. Rollouts update them with one compiled call per env step; learners fold the pure
``apply`` functions (with ``stats``) into their own compiled batch preparation.
Network-internal normalization belongs to the model construction adapter.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _check_batch(xs, means):
    if not isinstance(xs, dict) or xs.keys() != means.keys():
        raise ValueError("Observation batches must match the running-statistics keys")
    batch_sizes = set()
    for key, mean in means.items():
        shape = np.shape(xs[key])
        if len(shape) != mean.ndim + 1 or shape[1:] != mean.shape:
            raise ValueError(f"Observation batch {key!r} must have shape (batch, {mean.shape})")
        batch_sizes.add(shape[0])
    if len(batch_sizes) != 1 or 0 in batch_sizes:
        raise ValueError("Observation batches must share a positive leading dimension")


def _update_moments(xs, means, variances, count):
    batch_count = next(iter(xs.values())).shape[0]
    total_count = count + batch_count
    next_means = {}
    next_variances = {}
    for key, mean in means.items():
        x = xs[key].astype(mean.dtype)
        delta = jnp.mean(x, axis=0) - mean
        next_means[key] = mean + delta * batch_count / total_count
        next_variances[key] = (
            variances[key] * count
            + jnp.var(x, axis=0) * batch_count
            + jnp.square(delta) * count * batch_count / total_count
        ) / total_count
    return next_means, next_variances, total_count


def _apply(xs, stats):
    means, variances = stats
    return {key: (xs[key] - mean) / jnp.sqrt(variances[key] + 1e-8) for key, mean in means.items()}


def _apply_empirical(xs, stats):
    means, variances = stats
    return {
        key: (value - means[key]) / (jnp.sqrt(variances[key]) + 0.01) for key, value in xs.items()
    }


@jax.jit
def _observe(xs, means, variances, count, frozen_stats):
    means, variances, count = _update_moments(xs, means, variances, count)
    stats = (means, variances) if frozen_stats is None else frozen_stats
    return means, variances, count, _apply(xs, stats)


@jax.jit
def _observe_empirical(next_xs, xs, means, variances, count):
    next_xs = _apply_empirical(next_xs, (means, variances))
    means, variances, count = _update_moments(xs, means, variances, count)
    return means, variances, count, next_xs, _apply_empirical(xs, (means, variances))


class RunningMeanStd:
    """Per-key running mean, variance and count of observation batches."""

    apply = staticmethod(_apply)
    apply_empirical = staticmethod(_apply_empirical)

    def __init__(self, epsilon=1e-4, shapes: dict | None = None):
        if shapes is None:
            shapes = {"unified_obs": ()}
        elif not isinstance(shapes, dict):
            raise TypeError("shapes must be a dict")
        self.means = {key: jnp.zeros(shape, dtype=jnp.float32) for key, shape in shapes.items()}
        self.vars = {key: jnp.ones(shape, dtype=jnp.float32) for key, shape in shapes.items()}
        self.count = jnp.asarray(epsilon, dtype=jnp.float32)

    @classmethod
    def _of(cls, means, variances, count) -> "RunningMeanStd":
        """Wrap existing device statistics without creating (transferring) new arrays."""
        instance = cls.__new__(cls)
        instance.means, instance.vars, instance.count = means, variances, count
        return instance

    @property
    def stats(self):
        return self.means, self.vars

    def snapshot(self) -> "RunningMeanStd":
        """Frozen copy sharing the current (immutable) device arrays; no transfer."""
        return RunningMeanStd._of(dict(self.means), dict(self.vars), self.count)

    def normalize(self, xs):
        return _normalize(jax.device_put(xs), self.stats)

    def update(self, xs):
        if not self.means:
            return
        _check_batch(xs, self.means)
        self.means, self.vars, self.count = _update(
            jax.device_put(xs), self.means, self.vars, self.count
        )

    def observe(self, xs, frozen: "RunningMeanStd | None" = None):
        """Update with ``xs`` and return ``xs`` normalized in the same compiled call.

        Normalization uses ``frozen`` statistics when given, else the updated ones.
        """
        _check_batch(xs, self.means)
        self.means, self.vars, self.count, xs = _observe(
            jax.device_put(xs),
            self.means,
            self.vars,
            self.count,
            None if frozen is None else frozen.stats,
        )
        return xs

    def to_state(self):
        means, variances, count = jax.device_get((self.means, self.vars, self.count))
        return {
            "means": {key: np.asarray(arr) for key, arr in means.items()},
            "vars": {key: np.asarray(arr) for key, arr in variances.items()},
            "count": np.asarray(count, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state):
        means, vars_ = state["means"], state["vars"]
        if not isinstance(means, dict) or not isinstance(vars_, dict):
            raise TypeError("Running statistics means and vars must be dictionaries")
        if means.keys() != vars_.keys():
            raise ValueError("Running statistics means and vars must have matching keys")
        if any(np.shape(means[key]) != np.shape(vars_[key]) for key in means):
            raise ValueError("Running statistics means and vars must have matching shapes")
        return cls._of(
            *jax.device_put(
                (
                    {key: np.asarray(arr, np.float32) for key, arr in means.items()},
                    {key: np.asarray(arr, np.float32) for key, arr in vars_.items()},
                    np.asarray(state["count"], np.float32),
                )
            )
        )


_update = jax.jit(_update_moments)
_normalize = jax.jit(_apply)
_normalize_empirical = jax.jit(_apply_empirical)


def normalize_empirical_observation(obs, rms: RunningMeanStd | None):
    """Model inputs with frozen empirical statistics; identity without statistics."""
    return obs if rms is None else _normalize_empirical(jax.device_put(obs), rms.stats)


def observe_empirical_observations(rms: RunningMeanStd | None, next_obs, obs):
    """On-policy rollout step in one compiled call.

    ``next_obs`` is normalized with the old statistics; ``obs`` then updates them and is
    normalized with the new ones. Identity without statistics.
    """
    if rms is None:
        return next_obs, obs
    _check_batch(obs, rms.means)
    rms.means, rms.vars, rms.count, next_obs, obs = _observe_empirical(
        *jax.device_put((next_obs, obs)), rms.means, rms.vars, rms.count
    )
    return next_obs, obs


def _record_inputs(all_active, rewards, terminated, truncated, active):
    """One step's rollout values on device.

    Host rollouts pack them into one array, so a step costs a single upload instead of one
    per value; device rollouts pass their arrays through.
    """
    expected = all_active.shape
    values = (rewards, terminated, truncated, all_active if active is None else active)
    if any((np.shape(value) or (1,)) != expected for value in values):
        raise ValueError(
            f"rewards, terminated, truncated and active must match the worker shape {expected}"
        )
    if isinstance(rewards, jax.Array):
        return jax.device_put(values)
    if active is None:
        values = (*values[:3], np.ones(expected, bool))
    return jax.device_put(
        np.stack([np.broadcast_to(np.asarray(value, np.float32), expected) for value in values])
    )


def _worker_flags(inputs):
    # `inputs` is a (rewards, terminated, truncated, active) tuple or their packed rows.
    rewards, terminated, truncated, active = inputs
    rewards = jnp.atleast_1d(rewards).astype(jnp.float32)
    dones = jnp.atleast_1d(terminated).astype(bool) | jnp.atleast_1d(truncated).astype(bool)
    return rewards, dones, jnp.atleast_1d(active).astype(bool)


@jax.jit(static_argnames=("gamma",))
def _record_returns(returns, mean, variance, count, inputs, gamma):
    rewards, dones, active = _worker_flags(inputs)
    returns = jnp.where(active, gamma * returns + rewards, returns)
    batch_count = jnp.sum(active)
    batch_mean = jnp.sum(jnp.where(active, returns, 0)) / jnp.maximum(batch_count, 1)
    batch_var = jnp.sum(jnp.where(active, jnp.square(returns - batch_mean), 0)) / jnp.maximum(
        batch_count, 1
    )
    total = count + batch_count
    delta = batch_mean - mean
    next_mean = mean + delta * batch_count / total
    next_variance = (
        variance * count + batch_var * batch_count + jnp.square(delta) * count * batch_count / total
    ) / total
    return jnp.where(active & dones, 0, returns), next_mean, next_variance, total


class RewardNormalizer:
    """Scales rewards by the running std of the discounted return (Engstrom et al. 2020).

    Each worker accumulates ``G_t = gamma * G_{t-1} + r_t`` during rollout; :meth:`record`
    feeds those returns into running moments, and :meth:`apply` divides sampled rewards by
    the running ``std(G)``, rescaling reward targets for fixed-support critics (XQC).
    """

    def __init__(self, worker_size: int, gamma: float):
        self.gamma = float(gamma)
        self.rms = RunningMeanStd(shapes={"return": ()})
        self.discounted_returns = jnp.zeros(int(worker_size), dtype=jnp.float32)
        self._all_active = jnp.ones(int(worker_size), dtype=bool)

    def record(self, rewards, terminated, truncated, active=None):
        self.discounted_returns, mean, variance, self.rms.count = _record_returns(
            self.discounted_returns,
            self.rms.means["return"],
            self.rms.vars["return"],
            self.rms.count,
            _record_inputs(self._all_active, rewards, terminated, truncated, active),
            gamma=self.gamma,
        )
        self.rms.means, self.rms.vars = {"return": mean}, {"return": variance}

    @property
    def stats(self):
        return self.rms.vars["return"]

    @staticmethod
    def apply(rewards, stats):
        return rewards.astype(jnp.float32) / jnp.sqrt(stats + 1e-8)

    @property
    def scale(self):
        """Current reward divisor (a device scalar; read it at logging time)."""
        return _reward_scale(self.stats)

    def normalize(self, rewards):
        return _normalize_rewards(jax.device_put(rewards), self.stats)

    def to_state(self):
        return self.rms.to_state()

    def reset(self):
        self.discounted_returns = _zeros_like(self.discounted_returns)

    def restore(self, state):
        self.rms = RunningMeanStd.from_state(state)
        self.reset()


_reward_scale = jax.jit(lambda variance: jnp.sqrt(variance + 1e-8))
# Compiled so the zero fill is part of the executable, not a host scalar sent at call time.
_zeros_like = jax.jit(jnp.zeros_like)
_normalize_rewards = jax.jit(RewardNormalizer.apply)


@jax.jit(static_argnames=("gamma",))
def _record_flashsac_returns(returns, mean, variance, count, maximum, inputs, gamma):
    rewards, dones, active = _worker_flags(inputs)
    returns = jnp.where(active, gamma * (1 - dones) * returns + rewards, returns)
    size = jnp.sum(active)
    batch_mean = jnp.sum(jnp.where(active, returns, 0)) / jnp.maximum(size, 1)
    batch_var = jnp.sum(jnp.where(active, (returns - batch_mean) ** 2, 0)) / jnp.maximum(size, 1)
    total = count + size
    ratio = size / jnp.maximum(total, 1)
    delta = batch_mean - mean
    # The reference adds epsilon to the old variance weight on every update.
    next_var = (
        variance * (count + 1e-4) + batch_var * size + delta**2 * count * ratio
    ) / jnp.maximum(total, 1)
    return (
        returns,
        jnp.where(size > 0, mean + delta * ratio, mean),
        jnp.where(size > 0, next_var, variance),
        total,
        jnp.maximum(maximum, jnp.max(jnp.where(active, jnp.abs(returns), 0))),
    )


def _flashsac_scale(stats, normalized_G_max):
    variance, maximum = stats
    return jnp.maximum(jnp.sqrt(variance + 1e-8), maximum / normalized_G_max)


class FlashSACRewardNormalizer(RewardNormalizer):
    """FlashSAC's divisor: ``max(std(G), max|G| / normalized_G_max)``."""

    def __init__(self, worker_size: int, gamma: float, normalized_G_max: float = 5.0):
        if worker_size < 1 or not 0 <= gamma <= 1:
            raise ValueError("worker_size must be positive and gamma must be in [0, 1]")
        if not math.isfinite(normalized_G_max) or normalized_G_max <= 0:
            raise ValueError("normalized_G_max must be finite and positive")
        super().__init__(worker_size, gamma)
        self.rms = RunningMeanStd(epsilon=0, shapes={"return": ()})
        self.normalized_G_max = float(normalized_G_max)
        self.max_abs_return = jnp.zeros((), dtype=jnp.float32)
        self._scale = jax.jit(partial(_flashsac_scale, normalized_G_max=self.normalized_G_max))
        self._normalize = jax.jit(self.apply)

    def record(self, rewards, terminated, truncated, active=None):
        (
            self.discounted_returns,
            mean,
            variance,
            self.rms.count,
            self.max_abs_return,
        ) = _record_flashsac_returns(
            self.discounted_returns,
            self.rms.means["return"],
            self.rms.vars["return"],
            self.rms.count,
            self.max_abs_return,
            _record_inputs(self._all_active, rewards, terminated, truncated, active),
            gamma=self.gamma,
        )
        self.rms.means, self.rms.vars = {"return": mean}, {"return": variance}

    @property
    def stats(self):
        return self.rms.vars["return"], self.max_abs_return

    def apply(self, rewards, stats):
        """Bound, not static: ``normalized_G_max`` is part of the compiled function."""
        return rewards.astype(jnp.float32) / _flashsac_scale(stats, self.normalized_G_max)

    @property
    def scale(self):
        return self._scale(self.stats)

    def normalize(self, rewards):
        return self._normalize(jax.device_put(rewards), self.stats)

    def to_state(self):
        return {
            **super().to_state(),
            "max_abs_return": np.asarray(jax.device_get(self.max_abs_return)),
        }

    def restore(self, state):
        maximum = np.asarray(state["max_abs_return"])
        if maximum.shape != () or not np.isfinite(maximum) or maximum < 0:
            raise ValueError("max_abs_return must be a finite nonnegative scalar")
        super().restore(state)
        self.max_abs_return = jax.device_put(maximum.astype(np.float32))
