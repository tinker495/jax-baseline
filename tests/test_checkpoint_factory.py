"""Contract tests for make_checkpoint_controller / CheckpointSetup.

Verifies the baseline_q default and explicit pass-through values, and that the
controller is wired with the resolved values.  No JAX import needed —
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
