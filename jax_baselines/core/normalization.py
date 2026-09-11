"""Observation and reward normalization state shared by algorithm families.

Rollouts record samples; learners and evaluation apply frozen statistics.
Network-internal normalization belongs to the model construction adapter.
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _update_device_moments(xs, means, variances, count):
    batch_count = next(iter(xs.values())).shape[0]
    total_count = count + batch_count
    next_means = {}
    next_variances = {}
    for key, mean in means.items():
        x = jnp.asarray(xs[key], dtype=mean.dtype)
        delta = jnp.mean(x, axis=0) - mean
        next_means[key] = mean + delta * batch_count / total_count
        next_variances[key] = (
            variances[key] * count
            + jnp.var(x, axis=0) * batch_count
            + jnp.square(delta) * count * batch_count / total_count
        ) / total_count
    return next_means, next_variances, total_count


@jax.jit
def _normalize_device_observations(xs, means, variances):
    return {key: (xs[key] - mean) / jnp.sqrt(variances[key] + 1e-8) for key, mean in means.items()}


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(
        self, epsilon=1e-4, shapes: dict | None = None, dtype=np.float64, *, on_device: bool = False
    ):
        """Track running statistics; device mode keeps all runtime state in float32 JAX arrays."""
        if shapes is None:
            shapes = {"unified_obs": ()}
        elif not isinstance(shapes, dict):
            raise TypeError("shapes must be a dict")
        self.on_device = on_device
        self.dtype = np.dtype(np.float32 if on_device else dtype)
        array_module = jnp if on_device else np
        self.means = {
            key: array_module.zeros(shape, dtype=self.dtype) for key, shape in shapes.items()
        }
        self.vars = {
            key: array_module.ones(shape, dtype=self.dtype) for key, shape in shapes.items()
        }
        self.count = jnp.asarray(epsilon, dtype=jnp.float32) if on_device else epsilon

    def normalize(self, xs):
        """Normalizes the input using the running mean and variance."""
        if self.on_device:
            return _normalize_device_observations(xs, self.means, self.vars)
        return {
            key: (np.asarray(xs[key]) - self.means[key]) / np.sqrt(self.vars[key] + 1e-8)
            for key in self.means
        }

    def update(self, xs):
        """Updates the mean, var and count from a batch of samples."""
        if self.on_device:
            if not isinstance(xs, dict) or xs.keys() != self.means.keys():
                raise ValueError("Observation batches must match the running-statistics keys")
            if not self.means:
                return
            batch_sizes = set()
            for key, mean in self.means.items():
                x = xs[key]
                if x.ndim != mean.ndim + 1 or x.shape[1:] != mean.shape:
                    raise ValueError(
                        f"Observation batch {key!r} must have shape (batch, {mean.shape})"
                    )
                batch_sizes.add(x.shape[0])
            if len(batch_sizes) != 1 or 0 in batch_sizes:
                raise ValueError("Observation batches must share a positive leading dimension")
            self.means, self.vars, self.count = _update_device_moments(
                xs, self.means, self.vars, self.count
            )
            return
        means = {}
        vars = {}
        batch_count = None
        for key, mean in self.means.items():
            x = np.asarray(xs[key])
            var = self.vars[key]
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            current_count = x.shape[0]
            if batch_count is not None and current_count != batch_count:
                raise ValueError("Observation batches must share a leading dimension")
            batch_count = current_count
            mean, var = self.update_mean_var_count_from_moments(
                mean, var, batch_mean, batch_var, batch_count
            )
            means[key] = mean
            vars[key] = var
        self.means = means
        self.vars = vars
        if batch_count is not None:
            self.count += batch_count

    def update_mean_var_count_from_moments(self, mean, var, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean

        tot_count = self.count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def to_state(self):
        """Serialize running statistics to a numpy-friendly state."""
        means, variances, count = jax.device_get((self.means, self.vars, self.count))
        return {
            "means": {key: np.asarray(arr) for key, arr in means.items()},
            "vars": {key: np.asarray(arr) for key, arr in variances.items()},
            "count": np.asarray(count, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state, *, on_device: bool = False):
        """Deserialize running statistics from a saved state."""
        if not on_device:
            state = jax.device_get(state)
        means = state["means"]
        vars_ = state["vars"]
        if not isinstance(means, dict) or not isinstance(vars_, dict):
            raise TypeError("Running statistics means and vars must be dictionaries")
        if means.keys() != vars_.keys():
            raise ValueError("Running statistics means and vars must have matching keys")
        array_module = jnp if on_device else np
        means = {key: array_module.asarray(arr) for key, arr in means.items()}
        vars_ = {key: array_module.asarray(arr) for key, arr in vars_.items()}
        if any(means[key].shape != vars_[key].shape for key in means):
            raise ValueError("Running statistics means and vars must have matching shapes")
        dtype = next(iter(means.values())).dtype if means else np.float64
        shapes = {key: arr.shape for key, arr in means.items()}
        instance = cls(shapes=shapes, dtype=dtype, on_device=on_device)
        instance.means = {
            key: array_module.asarray(arr, dtype=instance.dtype) for key, arr in means.items()
        }
        instance.vars = {
            key: array_module.asarray(arr, dtype=instance.dtype) for key, arr in vars_.items()
        }
        instance.count = (
            jnp.asarray(state["count"], dtype=jnp.float32)
            if on_device
            else float(np.asarray(state["count"]))
        )
        return instance


@jax.jit
def _normalize_empirical_device_observation(obs, means, variances):
    return {
        key: (value - means[key]) / (jnp.sqrt(variances[key]) + 0.01) for key, value in obs.items()
    }


def normalize_empirical_observation(obs, rms: RunningMeanStd | None, *, on_device: bool):
    """Prepare model inputs using frozen, per-key empirical observation statistics."""
    if not on_device:
        if rms is None:
            return {key: np.asarray(value) for key, value in obs.items()}
        return {
            key: np.asarray(
                (np.asarray(value) - rms.means[key]) / (np.sqrt(rms.vars[key]) + 0.01),
                dtype=np.float32,
            )
            for key, value in obs.items()
        }
    if rms is None:
        return obs
    return _normalize_empirical_device_observation(obs, rms.means, rms.vars)


@jax.jit(static_argnames=("gamma",))
def _record_device_returns(returns, means, variances, count, rewards, dones, active, gamma):
    returns = jnp.where(active, gamma * returns + rewards, returns)
    batch_count = jnp.sum(active)
    batch_mean = jnp.sum(jnp.where(active, returns, 0)) / jnp.maximum(batch_count, 1)
    batch_var = jnp.sum(jnp.where(active, jnp.square(returns - batch_mean), 0)) / jnp.maximum(
        batch_count, 1
    )
    total = count + batch_count
    delta = batch_mean - means["return"]
    mean = means["return"] + delta * batch_count / total
    variance = (
        variances["return"] * count
        + batch_var * batch_count
        + jnp.square(delta) * count * batch_count / total
    ) / total
    return jnp.where(active & dones, 0, returns), {"return": mean}, {"return": variance}, total


@jax.jit
def _normalize_device_rewards(rewards, variance):
    return rewards / jnp.sqrt(variance + 1e-8)


class RewardNormalizer:
    """Scales rewards by the running std of the discounted return (Engstrom et al. 2020).

    Each worker accumulates ``G_t = gamma * G_{t-1} + r_t`` during rollout;
    :meth:`record` feeds those returns into a :class:`RunningMeanStd`, and
    :meth:`normalize` divides sampled rewards by :attr:`scale` (the running
    ``std(G)``), rescaling reward targets for fixed-support critics (XQC).
    """

    def __init__(self, worker_size: int, gamma: float, *, on_device: bool = False):
        self.gamma = float(gamma)
        self.rms = RunningMeanStd(shapes={"return": ()}, dtype=np.float64, on_device=on_device)
        self.discounted_returns = (jnp if on_device else np).zeros(
            int(worker_size), dtype=self.rms.dtype
        )
        self._all_active = (jnp if on_device else np).ones(int(worker_size), dtype=bool)

    def record(self, rewards, dones, active=None):
        array_module = jnp if self.rms.on_device else np
        rewards = array_module.atleast_1d(array_module.asarray(rewards, dtype=self.rms.dtype))
        dones = array_module.atleast_1d(array_module.asarray(dones, dtype=bool))
        active = self._all_active if active is None else array_module.asarray(active, dtype=bool)
        expected_shape = self.discounted_returns.shape
        if rewards.shape != expected_shape or dones.shape != expected_shape:
            raise ValueError(
                "rewards and dones must match the configured worker shape "
                f"{expected_shape}, got {rewards.shape} and {dones.shape}"
            )
        if active.shape != expected_shape:
            raise ValueError(
                "active must match the configured worker shape "
                f"{expected_shape}, got {active.shape}"
            )
        if self.rms.on_device:
            (
                self.discounted_returns,
                self.rms.means,
                self.rms.vars,
                self.rms.count,
            ) = _record_device_returns(
                self.discounted_returns,
                self.rms.means,
                self.rms.vars,
                self.rms.count,
                rewards,
                dones,
                active,
                self.gamma,
            )
            return
        if not np.any(active):
            return

        self.discounted_returns[active] = (
            self.gamma * self.discounted_returns[active] + rewards[active]
        )
        self.rms.update({"return": self.discounted_returns[active]})
        self.discounted_returns[active & dones] = 0.0

    @property
    def scale(self):
        """Current reward divisor: std of the recorded discounted returns."""
        if self.rms.on_device:
            return jnp.sqrt(self.rms.vars["return"] + 1e-8)
        return float(np.sqrt(self.rms.vars["return"] + 1e-8))

    def normalize(self, rewards):
        if self.rms.on_device:
            return _normalize_device_rewards(rewards, self.rms.vars["return"])
        rewards = np.asarray(rewards)
        dtype = np.result_type(rewards.dtype, np.float32)
        return (rewards / self.scale).astype(dtype, copy=False)

    def to_state(self):
        return self.rms.to_state()

    def reset(self):
        if self.rms.on_device:
            self.discounted_returns = jnp.zeros_like(self.discounted_returns)
        else:
            self.discounted_returns = np.zeros(self.discounted_returns.shape, dtype=self.rms.dtype)

    def restore(self, state):
        self.rms = RunningMeanStd.from_state(state, on_device=self.rms.on_device)
        self.reset()
