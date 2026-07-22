import numpy as np


def compute_ckpt_window_stat(
    returns_window: list, q: float, use_standardization: bool, mode: str = "quantile"
):
    """Compute robust window statistic for checkpoint decisions.

    Args:
        returns_window: List of episode returns in current window
        q: Quantile parameter (used when mode="quantile")
        use_standardization: Whether to standardize using median and MAD
        mode: Baseline computation mode - "min", "median", "quantile", or "mean"

    Returns:
        Window statistic value or None if window is empty
    """
    if returns_window is None or len(returns_window) == 0:
        return None
    arr = np.asarray(returns_window, dtype=np.float64)
    if use_standardization and arr.size >= 3:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        scale = mad if mad > 1e-8 else 1.0
        arr = (arr - med) / scale

    # Dispatch by mode
    if mode == "min":
        return float(np.min(arr))
    elif mode == "median":
        return float(np.median(arr))
    elif mode == "mean":
        return float(np.mean(arr))
    elif mode in ("quantile", "lower_percent"):
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(arr, q))
    else:
        raise ValueError(
            f"Unsupported compute_ckpt_window_stat mode '{mode}'. "
            "Expected one of {'min', 'median', 'mean', 'quantile', 'lower_percent'}."
        )


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shapes: dict | None = None, dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        if shapes is None:
            shapes = {"obs": ()}
        elif not isinstance(shapes, dict):
            raise TypeError("shapes must be a dict")
        self.dtype = np.dtype(dtype)
        self.means = {key: np.zeros(shape, dtype=self.dtype) for key, shape in shapes.items()}
        self.vars = {key: np.ones(shape, dtype=self.dtype) for key, shape in shapes.items()}
        self.count = epsilon

    def normalize(self, xs):
        """Normalizes the input using the running mean and variance."""
        return {
            key: (xs[key] - self.means[key]) / np.sqrt(self.vars[key] + 1e-8) for key in self.means
        }

    def update(self, xs):
        """Updates the mean, var and count from a batch of samples."""
        means = {}
        vars = {}
        batch_count = None
        for key, mean in self.means.items():
            x = xs[key]
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
        return {
            "means": {key: np.asarray(arr) for key, arr in self.means.items()},
            "vars": {key: np.asarray(arr) for key, arr in self.vars.items()},
            "count": np.asarray(self.count, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state):
        """Deserialize running statistics from a saved state."""
        means = state.get("means", {})
        vars_ = state.get("vars", {})
        if not isinstance(means, dict):
            keys = ["obs"] if len(means) == 1 else [str(index) for index in range(len(means))]
            means = dict(zip(keys, means))
            vars_ = dict(zip(keys, vars_))
        means = {key: np.asarray(arr) for key, arr in means.items()}
        vars_ = {key: np.asarray(arr) for key, arr in vars_.items()}
        dtype = next(iter(means.values())).dtype if means else np.float64
        shapes = {key: arr.shape for key, arr in means.items()}
        instance = cls(shapes=shapes, dtype=dtype)
        if means:
            instance.means = {key: arr.astype(dtype, copy=False) for key, arr in means.items()}
        if vars_:
            instance.vars = {key: arr.astype(dtype, copy=False) for key, arr in vars_.items()}
        count = state.get("count", np.array(0.0))
        instance.count = float(np.asarray(count))
        return instance


class RewardNormalizer:
    """Scales rewards by the running std of the discounted return (Engstrom et al. 2020).

    Each worker accumulates ``G_t = gamma * G_{t-1} + r_t`` during rollout;
    :meth:`record` feeds those returns into a :class:`RunningMeanStd`, and
    :meth:`normalize` divides sampled rewards by :attr:`scale` (the running
    ``std(G)``), effectively bounding Q-values for fixed-support critics (XQC).
    """

    def __init__(self, worker_size: int, gamma: float):
        self.gamma = float(gamma)
        self.rms = RunningMeanStd(shapes={"return": ()}, dtype=np.float64)
        self.discounted_returns = np.zeros(int(worker_size), dtype=np.float64)

    def record(self, rewards, dones, active=None):
        rewards = np.atleast_1d(np.asarray(rewards, dtype=np.float64))
        dones = np.atleast_1d(np.asarray(dones, dtype=bool))
        active = (
            np.ones(rewards.shape, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        )
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
        return float(np.sqrt(self.rms.vars["return"] + 1e-8))

    def normalize(self, rewards):
        rewards = np.asarray(rewards)
        dtype = np.result_type(rewards.dtype, np.float32)
        return (rewards / self.scale).astype(dtype, copy=False)

    def to_state(self):
        return self.rms.to_state()

    def reset(self):
        self.discounted_returns.fill(0.0)

    def restore(self, state):
        self.rms = RunningMeanStd.from_state(state)
        self.reset()
