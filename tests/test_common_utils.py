"""Regression coverage for pure-numpy utilities in jax_baselines.math.statistics.

RunningMeanStd (Welford online stats) and compute_ckpt_window_stat are public
contracts consumed by the DQN/DDPG base classes (obs normalization and
checkpoint-gating). They had no direct test; these lock their behavior.
"""

import numpy as np
import pytest

from jax_baselines.math.statistics import (
    RewardNormalizer,
    RunningMeanStd,
    compute_ckpt_window_stat,
)


def test_reward_normalizer_tracks_active_discounted_returns_without_episode_bleed():
    normalizer = RewardNormalizer(worker_size=2, gamma=0.5)

    normalizer.record(
        rewards=np.array([2.0, 10.0]),
        dones=np.array([False, True]),
    )
    normalizer.record(
        rewards=np.array([4.0, 99.0]),
        dones=np.array([True, False]),
        active=np.array([True, False]),
    )

    np.testing.assert_array_equal(normalizer.discounted_returns, np.zeros(2))
    assert normalizer.rms.count == pytest.approx(3.0001)
    assert normalizer.rms.means["return"] == pytest.approx(17.0 / 3.0001)


def test_reward_normalizer_scale_is_return_std_and_divides_rewards():
    normalizer = RewardNormalizer(worker_size=1, gamma=0.5)
    normalizer.record(rewards=np.array([2.0]), dones=np.array([False]))
    normalizer.record(rewards=np.array([6.0]), dones=np.array([True]))

    assert normalizer.scale == pytest.approx(float(np.sqrt(normalizer.rms.vars["return"] + 1e-8)))
    np.testing.assert_allclose(
        normalizer.normalize(np.array([4.0, -4.0])),
        np.array([4.0, -4.0]) / normalizer.scale,
    )


def test_reward_normalizer_normalize_preserves_float32_dtype():
    normalizer = RewardNormalizer(worker_size=1, gamma=0.99)
    normalizer.record(rewards=np.array([5.0]), dones=np.array([False]))

    assert normalizer.normalize(np.ones((4, 1), dtype=np.float32)).dtype == np.float32
    assert normalizer.normalize(np.ones(3)).dtype == np.float64


def test_running_mean_std_matches_batch_statistics():
    rng = np.random.default_rng(0)
    batches = [rng.normal(loc=2.0, scale=3.0, size=(50, 4)) for _ in range(5)]
    rms = RunningMeanStd(shapes={"obs": (4,)})
    for batch in batches:
        rms.update({"obs": batch})

    allx = np.concatenate(batches, axis=0)
    # epsilon init (1e-4) makes the online estimate approach the exact moments
    assert np.allclose(rms.means["obs"], allx.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.vars["obs"], allx.var(axis=0), atol=1e-2)


def test_running_mean_std_normalize_centers_and_scales():
    rms = RunningMeanStd(shapes={"obs": (3,)})
    rms.means = {"obs": np.array([1.0, 2.0, 3.0])}
    rms.vars = {"obs": np.array([4.0, 4.0, 4.0])}
    out = rms.normalize({"obs": np.array([[1.0, 2.0, 3.0]])})["obs"]
    assert np.allclose(out, 0.0, atol=1e-3)


def test_running_mean_std_state_round_trip():
    rng = np.random.default_rng(1)
    rms = RunningMeanStd(shapes={"obs": (2,)})
    rms.update({"obs": rng.normal(size=(10, 2))})

    restored = RunningMeanStd.from_state(rms.to_state())
    assert np.array_equal(restored.means["obs"], rms.means["obs"])
    assert np.array_equal(restored.vars["obs"], rms.vars["obs"])
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
