"""Round-trip tests for the typed DPG checkpoint state.

These exercise the per-algorithm checkpoint *contract* (``checkpoint_params`` /
``load_checkpoint_params``) and the family-wide ``CheckpointState`` spine in
isolation -- each agent is built with ``__new__`` so no env / buffer / model is
needed. This is the test surface the typed module exists to provide; the former
``getattr``/``hasattr`` marshalling had none.
"""

import tempfile

import jax
import numpy as np
import pytest

from jax_baselines.common.checkpoint import make_checkpoint_controller
from jax_baselines.common.statistics import RunningMeanStd
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.SAC.sac import SAC
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TQC.tqc import TQC

# The exact network bundle each algorithm's checkpoint contract carries.
ALGO_FIELDS = {
    DDPG: ["policy_params", "critic_params", "target_policy_params", "target_critic_params"],
    TD3: ["policy_params", "critic_params", "target_policy_params", "target_critic_params"],
    SAC: ["policy_params", "critic_params", "target_critic_params", "log_ent_coef"],
    TQC: ["policy_params", "critic_params", "target_critic_params", "log_ent_coef"],
    CrossQ: ["policy_params", "critic_params", "log_ent_coef"],
    TD7: [
        "encoder_params",
        "policy_params",
        "critic_params",
        "fixed_encoder_params",
        "fixed_encoder_target_params",
        "target_policy_params",
        "target_critic_params",
    ],
}


def _controller():
    return make_checkpoint_controller(
        use_checkpointing=True,
        steps_before_checkpointing=10,
        max_eps_before_checkpointing=5,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="min",
        ckpt_baseline_q=None,
        ckpt_gate_mode=None,
        ckpt_gate_q=None,
        snapshot=lambda: None,
        log_metric=lambda *a, **k: None,
    ).controller


def _tree(scale):
    return {
        "dense": {
            "w": np.full((3, 2), scale, np.float32),
            "b": np.arange(2, dtype=np.float32) + scale,
        }
    }


def _field_value(name, i):
    # log_ent_coef is a scalar; every other field is a network-param pytree.
    return np.float32(-1.5) if name == "log_ent_coef" else _tree(float(i + 1))


def _assert_tree_equal(a, b):
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


@pytest.mark.parametrize("cls", list(ALGO_FIELDS), ids=lambda c: c.__name__)
def test_checkpoint_round_trip(cls):
    fields = ALGO_FIELDS[cls]

    src = cls.__new__(cls)
    src.simba = False
    src._checkpoint = _controller()
    src.train_steps_count = 11
    src._ckpt_update_residual = 2.5
    src.eval_snapshot = {"encoder": _tree(7.0), "policy": _tree(8.0)}
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))
    # Make the schedule state non-default so the spine must carry it.
    src._checkpoint._enabled = True
    src._checkpoint._update_count = 4
    src._checkpoint._baseline = 1.25

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = cls.__new__(cls)
        dst.simba = False
        dst._checkpoint = _controller()
        dst.load_params(d)

    # spine
    assert dst.train_steps_count == 11
    assert abs(dst._ckpt_update_residual - 2.5) < 1e-6
    assert dst._checkpoint._enabled is True
    assert dst._checkpoint._update_count == 4
    assert abs(dst._checkpoint._baseline - 1.25) < 1e-6
    _assert_tree_equal(src.eval_snapshot, dst.eval_snapshot)

    # per-algorithm bundle
    for name in fields:
        _assert_tree_equal(getattr(src, name), getattr(dst, name))

    # the self.params / self.target_params overloading is gone
    assert not hasattr(dst, "params")
    assert not hasattr(dst, "target_params")


def test_simba_obs_rms_round_trip():
    fields = ALGO_FIELDS[DDPG]

    src = DDPG.__new__(DDPG)
    src._checkpoint = _controller()
    src.train_steps_count = 3
    src._ckpt_update_residual = 0.0
    src.eval_snapshot = None
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))
    src.simba = True
    src.obs_rms = RunningMeanStd(shapes=[(2,)], dtype=np.float64)
    src.obs_rms.update([np.ones((4, 2), np.float64)])
    src.action_obs_rms = RunningMeanStd.from_state(src.obs_rms.to_state())
    src.checkpoint_obs_rms = None  # exercises the None branch

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = DDPG.__new__(DDPG)
        dst.simba = True
        dst._checkpoint = _controller()
        dst.load_params(d)

    assert dst.obs_rms is not None
    assert dst.action_obs_rms is not None
    assert dst.checkpoint_obs_rms is None
    assert dst.eval_snapshot is None
    _assert_tree_equal(src.obs_rms.to_state(), dst.obs_rms.to_state())


def test_base_contract_is_abstract():
    from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family

    base = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    with pytest.raises(NotImplementedError):
        base.checkpoint_params()
    with pytest.raises(NotImplementedError):
        base.load_checkpoint_params(object())
