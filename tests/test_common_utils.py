"""Regression coverage for pure-numpy utilities in jax_baselines.common.utils.

RunningMeanStd (Welford online stats) and compute_ckpt_window_stat are public
contracts consumed by the DQN/DDPG base classes (obs normalization and
checkpoint-gating). They had no direct test; these lock their behavior.
"""

import numpy as np
import pytest

from jax_baselines.common.utils import RunningMeanStd, compute_ckpt_window_stat


def test_running_mean_std_matches_batch_statistics():
    rng = np.random.default_rng(0)
    batches = [rng.normal(loc=2.0, scale=3.0, size=(50, 4)) for _ in range(5)]
    rms = RunningMeanStd(shapes=[(4,)])
    for batch in batches:
        rms.update([batch])

    allx = np.concatenate(batches, axis=0)
    # epsilon init (1e-4) makes the online estimate approach the exact moments
    assert np.allclose(rms.means[0], allx.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.vars[0], allx.var(axis=0), atol=1e-2)


def test_running_mean_std_normalize_centers_and_scales():
    rms = RunningMeanStd(shapes=[(3,)])
    rms.means = [np.array([1.0, 2.0, 3.0])]
    rms.vars = [np.array([4.0, 4.0, 4.0])]
    out = rms.normalize([np.array([[1.0, 2.0, 3.0]])])[0]
    assert np.allclose(out, 0.0, atol=1e-3)


def test_running_mean_std_state_round_trip():
    rng = np.random.default_rng(1)
    rms = RunningMeanStd(shapes=[(2,)])
    rms.update([rng.normal(size=(10, 2))])

    restored = RunningMeanStd.from_state(rms.to_state())
    assert np.array_equal(restored.means[0], rms.means[0])
    assert np.array_equal(restored.vars[0], rms.vars[0])
    assert restored.count == pytest.approx(rms.count)


def test_compute_ckpt_window_stat_modes():
    window = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert compute_ckpt_window_stat(window, q=0.5, use_standardization=False, mode="min") == 0.0
    assert compute_ckpt_window_stat(window, q=0.5, use_standardization=False, mode="median") == 2.0
    assert compute_ckpt_window_stat(window, q=0.5, use_standardization=False, mode="mean") == 2.0
    assert (
        compute_ckpt_window_stat(window, q=0.0, use_standardization=False, mode="quantile") == 0.0
    )


def test_compute_ckpt_window_stat_empty_returns_none():
    assert compute_ckpt_window_stat([], q=0.5, use_standardization=False) is None
    assert compute_ckpt_window_stat(None, q=0.5, use_standardization=False) is None


def test_compute_ckpt_window_stat_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        compute_ckpt_window_stat([1.0, 2.0], q=0.5, use_standardization=False, mode="nope")
