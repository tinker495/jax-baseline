"""Contract tests for make_checkpoint_controller / CheckpointSetup.

Verifies the gate_q resolution ladder exactly, baseline_q default,
gate_mode default, and explicit pass-through values.  No JAX import needed —
checkpoint.py is pure Python + numpy.
"""

from jax_baselines.common.checkpoint import (
    CheckpointController,
    CheckpointSetup,
    make_checkpoint_controller,
)


def _noop(*_):
    pass


def _make(**overrides):
    """Call make_checkpoint_controller with minimal defaults + overrides."""
    defaults = dict(
        use_checkpointing=False,
        steps_before_checkpointing=0,
        max_eps_before_checkpointing=1,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="mean",
        ckpt_baseline_q=None,
        ckpt_gate_mode=None,
        ckpt_gate_q=None,
        snapshot=_noop,
        log_metric=_noop,
    )
    defaults.update(overrides)
    return make_checkpoint_controller(**defaults)


# -- return type --


def test_returns_checkpoint_setup():
    setup = _make()
    assert isinstance(setup, CheckpointSetup)
    assert isinstance(setup.controller, CheckpointController)


# -- quantile default --


def test_quantile_is_0_2():
    setup = _make()
    assert setup.quantile == 0.2


# -- use_return_standardization default --


def test_use_return_standardization_is_false():
    setup = _make()
    assert setup.use_return_standardization is False


# -- baseline_q resolution --


def test_baseline_q_defaults_to_quantile_when_none():
    setup = _make(ckpt_baseline_q=None)
    assert setup.baseline_q == setup.quantile  # 0.2


def test_baseline_q_explicit_passes_through():
    setup = _make(ckpt_baseline_q=0.7)
    assert setup.baseline_q == 0.7


# -- gate_mode resolution --


def test_gate_mode_defaults_to_baseline_mode_when_none():
    setup = _make(ckpt_baseline_mode="min", ckpt_gate_mode=None)
    assert setup.gate_mode == "min"


def test_gate_mode_explicit_passes_through():
    setup = _make(ckpt_baseline_mode="min", ckpt_gate_mode="mean")
    assert setup.gate_mode == "mean"


# -- gate_q resolution ladder --


def test_gate_q_explicit_overrides_everything():
    setup = _make(ckpt_gate_mode="median", ckpt_gate_q=0.9)
    assert setup.gate_q == 0.9


def test_gate_q_median_mode_gives_0_5():
    setup = _make(ckpt_baseline_mode="median", ckpt_gate_mode=None, ckpt_gate_q=None)
    assert setup.gate_mode == "median"
    assert setup.gate_q == 0.5


def test_gate_q_median_explicit_mode_gives_0_5():
    setup = _make(ckpt_gate_mode="median", ckpt_gate_q=None)
    assert setup.gate_q == 0.5


def test_gate_q_quantile_mode_gives_baseline_q():
    baseline_q = 0.3
    setup = _make(ckpt_baseline_q=baseline_q, ckpt_gate_mode="quantile", ckpt_gate_q=None)
    assert setup.gate_q == baseline_q


def test_gate_q_min_mode_gives_baseline_q():
    baseline_q = 0.4
    setup = _make(ckpt_baseline_q=baseline_q, ckpt_gate_mode="min", ckpt_gate_q=None)
    assert setup.gate_q == baseline_q


def test_gate_q_mean_mode_gives_baseline_q():
    baseline_q = 0.6
    setup = _make(ckpt_baseline_q=baseline_q, ckpt_gate_mode="mean", ckpt_gate_q=None)
    assert setup.gate_q == baseline_q


def test_gate_q_unknown_mode_gives_quantile():
    setup = _make(ckpt_gate_mode="unknown_mode", ckpt_gate_q=None)
    assert setup.gate_q == setup.quantile  # 0.2


def test_gate_q_unknown_mode_with_custom_baseline_q_still_gives_quantile():
    # gate_q falls back to quantile (0.2), NOT baseline_q, for unrecognised modes
    setup = _make(ckpt_baseline_q=0.8, ckpt_gate_mode="custom", ckpt_gate_q=None)
    assert setup.gate_q == 0.2


# -- baseline_mode preserved on setup --


def test_baseline_mode_preserved():
    setup = _make(ckpt_baseline_mode="quantile")
    assert setup.baseline_mode == "quantile"


# -- controller wired with resolved values --


def test_controller_receives_resolved_baseline_q():
    setup = _make(ckpt_baseline_q=None)  # should resolve to 0.2
    assert setup.controller._baseline_q == 0.2


def test_controller_receives_explicit_baseline_q():
    setup = _make(ckpt_baseline_q=0.55)
    assert setup.controller._baseline_q == 0.55


def test_controller_receives_baseline_mode():
    setup = _make(ckpt_baseline_mode="min")
    assert setup.controller._baseline_mode == "min"
