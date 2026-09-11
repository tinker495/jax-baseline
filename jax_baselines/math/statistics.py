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
