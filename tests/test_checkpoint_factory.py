"""Contract tests for make_checkpoint_scaffold / CheckpointScaffold.

Verifies the baseline_q default and explicit pass-through values, that the
controller is wired with the resolved values (white-box, since the scaffold
delegates and exposes no public ``.controller``), the hparams() contract, and
the schedule delegation. No JAX import needed -- checkpoint.py is pure Python +
numpy.
"""

from jax_baselines.core.checkpoint import (
    CheckpointController,
    CheckpointScaffold,
    make_checkpoint_scaffold,
)


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
        log_metric=_noop,
    )
    defaults.update(overrides)
    return make_checkpoint_scaffold(**defaults)


# -- return type --


def test_returns_checkpoint_scaffold():
    scaffold = _make()
    assert isinstance(scaffold, CheckpointScaffold)
    assert isinstance(scaffold._controller, CheckpointController)


# -- baseline_q resolution (observed via hparams) --


def test_baseline_q_defaults_to_0_2_when_none():
    scaffold = _make(ckpt_baseline_q=None)
    assert scaffold.hparams()["baseline_q"] == 0.2


def test_baseline_q_explicit_passes_through():
    scaffold = _make(ckpt_baseline_q=0.7)
    assert scaffold.hparams()["baseline_q"] == 0.7


# -- baseline_mode preserved (observed via hparams) --


def test_baseline_mode_preserved():
    scaffold = _make(ckpt_baseline_mode="quantile")
    assert scaffold.hparams()["baseline_mode"] == "quantile"


# -- controller wired with resolved values (white-box wiring contract) --


def test_controller_receives_resolved_baseline_q():
    scaffold = _make(ckpt_baseline_q=None)  # should resolve to 0.2
    assert scaffold._controller._baseline_q == 0.2


def test_controller_receives_explicit_baseline_q():
    scaffold = _make(ckpt_baseline_q=0.55)
    assert scaffold._controller._baseline_q == 0.55


def test_controller_receives_baseline_mode():
    scaffold = _make(ckpt_baseline_mode="min")
    assert scaffold._controller._baseline_mode == "min"


# -- hparams provider contract --


def test_hparams_contract():
    scaffold = _make(
        use_checkpointing=True,
        steps_before_checkpointing=123,
        max_eps_before_checkpointing=7,
        ckpt_baseline_mode="median",
        ckpt_baseline_q=0.3,
    )
    hp = scaffold.hparams()
    assert set(hp) == {
        "use_checkpointing",
        "steps_before_checkpointing",
        "max_eps_before_checkpointing",
        "baseline_mode",
        "baseline_q",
    }
    assert hp == {
        "use_checkpointing": True,
        "steps_before_checkpointing": 123,
        "max_eps_before_checkpointing": 7,
        "baseline_mode": "median",
        "baseline_q": 0.3,
    }


# -- schedule delegation to the controller --


def test_delegates_to_controller():
    scaffold = _make(use_checkpointing=True)

    # read-only views forward
    assert scaffold.enabled == scaffold._controller.enabled
    assert scaffold.last_update_step == scaffold._controller.last_update_step

    # from_state mutates the controller; enabled reflects it through delegation
    scaffold.from_state({"checkpointing_enabled": True})
    assert scaffold._controller.enabled is True
    assert scaffold.enabled is True

    # to_state forwards the controller's serialized schedule state (keys + the
    # enabled flag we just set; dict == dict is avoided -- values are np arrays)
    state = scaffold.to_state()
    assert state.keys() == scaffold._controller.to_state().keys()
    assert bool(state["checkpointing_enabled"]) is True

    # on_episode_end forwards to the controller and returns its bool result.
    assert scaffold.on_episode_end(0, 1.0, 1) is True
