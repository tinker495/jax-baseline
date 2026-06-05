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

    def __init__(self, epsilon=1e-4, shapes: list = [()], dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.dtype = np.dtype(dtype)
        self.means = [np.zeros(shape, dtype=self.dtype) for shape in shapes]
        self.vars = [np.ones(shape, dtype=self.dtype) for shape in shapes]
        self.count = epsilon

    def normalize(self, xs):
        """Normalizes the input using the running mean and variance."""
        return [(x - mean) / np.sqrt(var + 1e-8) for x, mean, var in zip(xs, self.means, self.vars)]

    def update(self, xs):
        """Updates the mean, var and count from a batch of samples."""
        means = []
        vars = []
        for x, mean, var in zip(xs, self.means, self.vars):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            mean, var = self.update_mean_var_count_from_moments(
                mean, var, batch_mean, batch_var, batch_count
            )
            means.append(mean)
            vars.append(var)
        self.means = means
        self.vars = vars
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
            "means": [np.asarray(arr) for arr in self.means],
            "vars": [np.asarray(arr) for arr in self.vars],
            "count": np.asarray(self.count, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state):
        """Deserialize running statistics from a saved state."""
        means = [np.asarray(arr) for arr in state.get("means", [])]
        vars_ = [np.asarray(arr) for arr in state.get("vars", [])]
        dtype = means[0].dtype if means else np.float64
        shapes = [arr.shape for arr in means]
        instance = cls(shapes=shapes, dtype=dtype)
        if means:
            instance.means = [arr.astype(dtype, copy=False) for arr in means]
        if vars_:
            instance.vars = [arr.astype(dtype, copy=False) for arr in vars_]
        count = state.get("count", np.array(0.0))
        instance.count = float(np.asarray(count))
        return instance
