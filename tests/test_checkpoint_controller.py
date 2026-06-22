"""Contract tests for jax_baselines.core.checkpoint.CheckpointController.

The TD7-style checkpoint schedule was copy-pasted into the Q-Net and DPG base
classes and had drifted (the DPG copy fell through its warmup branch into a
second training pulse). It now lives in one deep module reachable without an
agent, env, or JAX. These lock the schedule: the disabled short-circuit, the
warmup early-return (Q-Net semantics, adopted as canonical), the enable
transition, end-of-window refresh, the below-baseline failure signal, and the
serialization round-trip the DPG family depends on.
"""

import numpy as np

from jax_baselines.core.checkpoint import CheckpointController


def make_controller(**overrides):
    """Build a controller with recording snapshot/log/pulse adapters."""
    calls = {"snapshot": 0, "pulse": [], "logs": []}

    config = dict(
        use_checkpointing=True,
        steps_before_checkpointing=0,
        max_eps_before_checkpointing=1,
        initial_window=1,
        baseline_q=0.2,
        baseline_mode="mean",
        use_return_standardization=False,
    )
    config.update(overrides)

    controller = CheckpointController(
        snapshot=lambda: calls.__setitem__("snapshot", calls["snapshot"] + 1),
        log_metric=lambda k, v, s: calls["logs"].append((k, v, s)),
        **config,
    )
    pulse = lambda steps, acc: calls["pulse"].append((steps, acc))  # noqa: E731
    return controller, calls, pulse


def test_disabled_short_circuits():
    controller, calls, pulse = make_controller(use_checkpointing=False)
    assert controller.on_episode_end(10, 5.0, 3, pulse) is True
    assert calls["snapshot"] == 0
    assert calls["pulse"] == []
    assert controller.enabled is False
    assert controller.last_update_step is None


def test_warmup_snapshots_and_returns_without_arming():
    # steps stay below the enable threshold -> warmup branch only.
    controller, calls, pulse = make_controller(steps_before_checkpointing=1000)
    result = controller.on_episode_end(10, 7.0, 4, pulse)
    assert result is True
    assert controller.enabled is False
    assert calls["snapshot"] == 1  # warmup snapshots a baseline...
    assert calls["pulse"] == [(10, 4)]  # ...and fires exactly one pulse
    assert controller.last_update_step is None  # but does not count as a refresh
    # window_stat is logged, baseline metric is not (no refresh)
    assert ("ckpt/window_stat", 7.0, 10) in calls["logs"]
    assert all(k != "ckpt/ckpt_baseline" for k, _, _ in calls["logs"])


def test_non_monitored_worker_feeds_pulse_volume_without_arming():
    # advance_criterion=False (a non-monitored vectorized worker) records its
    # timesteps toward the pulse volume but must not snapshot, fire, or advance
    # the assessment window. The monitored worker's later pulse then carries the
    # pooled timesteps (3 + 4 == 7).
    controller, calls, pulse = make_controller(steps_before_checkpointing=1000)

    assert controller.on_episode_end(10, 5.0, 3, pulse, advance_criterion=False) is True
    assert calls["snapshot"] == 0
    assert calls["pulse"] == []
    assert calls["logs"] == []

    controller.on_episode_end(20, 7.0, 4, pulse, advance_criterion=True)
    assert calls["snapshot"] == 1
    assert calls["pulse"] == [(20, 7)]


def test_enable_transition():
    controller, _, pulse = make_controller(steps_before_checkpointing=100)
    controller.on_episode_end(50, 1.0, 1, pulse)
    assert controller.enabled is False
    controller.on_episode_end(150, 1.0, 1, pulse)  # steps now past threshold
    assert controller.enabled is True


def test_refresh_then_below_baseline():
    # Enables immediately (steps_before=0); one episode per window (max_eps=1).
    controller, calls, pulse = make_controller(
        steps_before_checkpointing=0, max_eps_before_checkpointing=1
    )

    # First episode: enabled, baseline is -1e8, so window_stat refreshes it.
    assert controller.on_episode_end(10, 10.0, 5, pulse) is True
    assert calls["snapshot"] == 1
    assert controller.last_update_step == 10
    assert ("ckpt/ckpt_baseline", 10.0, 10) in calls["logs"]
    assert calls["pulse"][-1] == (10, 5)

    # Second episode below the refreshed baseline (10.0) -> failure signal.
    assert controller.on_episode_end(20, 2.0, 6, pulse) is False
    assert calls["snapshot"] == 1  # no new snapshot on failure
    assert calls["pulse"][-1] == (20, 6)  # but the pulse still fires


def test_window_accumulates_until_max_eps():
    # Two episodes per window: the first must not refresh.
    controller, calls, pulse = make_controller(
        steps_before_checkpointing=0, max_eps_before_checkpointing=2
    )
    assert controller.on_episode_end(10, 5.0, 3, pulse) is True
    assert calls["snapshot"] == 0  # window not full yet
    assert controller.last_update_step is None
    assert controller.on_episode_end(20, 5.0, 3, pulse) is True
    assert calls["snapshot"] == 1  # window full -> refresh
    assert controller.last_update_step == 20


def test_state_round_trip():
    src, _, pulse = make_controller(max_eps_before_checkpointing=2)
    src.on_episode_end(10, 5.0, 3, pulse)
    src.on_episode_end(20, 9.0, 4, pulse)  # drives enable + a refresh

    state = src.to_state()
    # keys are preserved verbatim for backward-compatible DPG checkpoints
    assert "checkpointing_enabled" in state
    assert "_ckpt_returns_window" in state

    dst, _, _ = make_controller(max_eps_before_checkpointing=2)
    dst.from_state(state)

    assert dst.enabled == src.enabled
    assert dst.last_update_step == src.last_update_step
    # full schedule state matches after a round trip
    for key, value in src.to_state().items():
        np.testing.assert_array_equal(np.asarray(dst.to_state()[key]), np.asarray(value))
