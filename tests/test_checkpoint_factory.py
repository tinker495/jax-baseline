"""Contract tests for the checkpoint controller factory.

Verifies default resolution, exposed configuration, and schedule behavior.
"""

from jax_baselines.core.checkpoint import CheckpointController, make_checkpoint_scaffold


def _noop(*_):
    pass


def _make(**overrides):
    """Call make_checkpoint_scaffold with minimal defaults + overrides."""
    defaults = dict(
        use_checkpointing=False,
        steps_before_checkpointing=0,
        max_eps_before_checkpointing=1,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="mean",
        ckpt_baseline_q=None,
        snapshot=_noop,
    )
    defaults.update(overrides)
    return make_checkpoint_scaffold(**defaults)


# -- return type --


def test_returns_checkpoint_controller():
    assert isinstance(_make(), CheckpointController)


# -- baseline_q resolution --


def test_baseline_q_defaults_to_0_2_when_none():
    controller = _make(ckpt_baseline_q=None)
    assert controller.baseline_q == 0.2


def test_baseline_q_explicit_passes_through():
    controller = _make(ckpt_baseline_q=0.7)
    assert controller.baseline_q == 0.7


# -- baseline_mode preserved --


def test_baseline_mode_preserved():
    controller = _make(ckpt_baseline_mode="quantile")
    assert controller.baseline_mode == "quantile"


def test_exposes_factory_configuration():
    controller = _make(
        use_checkpointing=True,
        steps_before_checkpointing=123,
        max_eps_before_checkpointing=7,
        ckpt_baseline_mode="median",
        ckpt_baseline_q=0.3,
    )
    assert controller.use_checkpointing is True
    assert controller.steps_before_checkpointing == 123
    assert controller.max_eps_before_checkpointing == 7
    assert controller.baseline_mode == "median"
    assert controller.baseline_q == 0.3


def test_controller_serializes_and_runs_schedule():
    controller = _make(use_checkpointing=True)
    controller.from_state({"checkpointing_enabled": True})
    assert controller.enabled is True

    state = controller.to_state()
    assert bool(state["checkpointing_enabled"]) is True
    assert controller.on_episode_end(0, 1.0, 1) is True
