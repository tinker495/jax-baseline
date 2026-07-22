"""Round-trip tests for the typed DPG warm-start checkpoint state.

These exercise the per-algorithm checkpoint *contract* (``checkpoint_params`` /
``load_checkpoint_params``) and the family-wide ``CheckpointState`` spine in
isolation -- each agent is built with ``__new__`` so no env / buffer / model is
needed. This is the test surface the typed module exists to provide; the former
``getattr``/``hasattr`` marshalling had none.

Optimizer, PRNG, replay, and environment state are intentionally outside this
warm-start contract.
"""

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.checkpoint_store import FileCheckpointStore
from jax_baselines.core.checkpoint import make_checkpoint_scaffold, snapshot_pytree
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.statistics import RewardNormalizer, RunningMeanStd
from jax_baselines.SAC.sac import SAC
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TQC.tqc import TQC
from jax_baselines.XQC.xqc import XQC

# The exact network bundle each algorithm's checkpoint contract carries.
ALGO_FIELDS = {
    DDPG: [
        "policy_params",
        "critic_params",
        "target_policy_params",
        "target_critic_params",
    ],
    TD3: [
        "policy_params",
        "critic_params",
        "target_policy_params",
        "target_critic_params",
    ],
    SAC: ["policy_params", "critic_params", "target_critic_params", "log_ent_coef"],
    TQC: ["policy_params", "critic_params", "target_critic_params", "log_ent_coef"],
    CrossQ: ["policy_params", "critic_params", "log_ent_coef"],
    XQC: ["policy_params", "critic_params", "target_critic_params", "log_ent_coef"],
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


def _scaffold():
    return make_checkpoint_scaffold(
        use_checkpointing=True,
        steps_before_checkpointing=10,
        max_eps_before_checkpointing=5,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="min",
        ckpt_baseline_q=None,
        snapshot=lambda: None,
    )


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


def test_snapshot_pytree_rebuilds_containers_without_copying_jax_array_leaves():
    jax_leaf = jnp.asarray([1.0, 2.0])
    numpy_leaf = np.asarray([3.0, 4.0])
    params = {"nested": {"jax": jax_leaf, "numpy": numpy_leaf}}

    snapshot = snapshot_pytree(params)

    assert snapshot is not params
    assert snapshot["nested"] is not params["nested"]
    assert snapshot["nested"]["jax"] is jax_leaf
    assert snapshot["nested"]["numpy"] is not numpy_leaf
    numpy_leaf[0] = 99.0
    np.testing.assert_array_equal(snapshot["nested"]["numpy"], np.asarray([3.0, 4.0]))


def test_family_checkpoint_snapshots_use_pytree_snapshot():
    qnet = Q_Network_Family.__new__(Q_Network_Family)
    qnet_params = {"policy": {"w": jnp.asarray([1.0]), "b": np.asarray([2.0])}}
    qnet.get_eval_params = lambda: qnet_params

    Q_Network_Family._checkpoint_update_snapshot(qnet)
    qnet_params["policy"]["w"] = jnp.asarray([9.0])
    qnet_params["policy"]["b"][0] = 9.0

    np.testing.assert_array_equal(np.asarray(qnet.checkpoint_params["policy"]["w"]), [1.0])
    np.testing.assert_array_equal(qnet.checkpoint_params["policy"]["b"], [2.0])

    dpg = DDPG.__new__(DDPG)
    dpg.simba = False
    eval_state = {"encoder": None, "policy": {"w": jnp.asarray([3.0])}}
    dpg.get_eval_state = lambda: eval_state

    DDPG._checkpoint_update_snapshot(dpg)
    eval_state["policy"]["w"] = jnp.asarray([7.0])

    np.testing.assert_array_equal(np.asarray(dpg.eval_snapshot["policy"]["w"]), [3.0])


@pytest.mark.parametrize("cls", list(ALGO_FIELDS), ids=lambda c: c.__name__)
def test_checkpoint_round_trip(cls):
    fields = ALGO_FIELDS[cls]

    src = cls.__new__(cls)
    src.checkpoint_store = FileCheckpointStore()
    src.simba = False
    src.reward_normalizer = None
    src.ckpt = _scaffold()
    src.train_steps_count = 11
    src._ckpt_update_residual = 2.5
    src.eval_snapshot = {"encoder": _tree(7.0), "policy": _tree(8.0)}
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))
    # Make the schedule state non-default so the spine must carry it.
    src.ckpt._enabled = True
    src.ckpt._update_count = 4
    src.ckpt._baseline = 1.25

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = cls.__new__(cls)
        dst.checkpoint_store = FileCheckpointStore()
        dst.simba = False
        dst.reward_normalizer = None
        dst.ckpt = _scaffold()
        dst.load_params(d)

    # spine
    assert dst.train_steps_count == 11
    assert abs(dst._ckpt_update_residual - 2.5) < 1e-6
    assert dst.ckpt._enabled is True
    assert dst.ckpt._update_count == 4
    assert abs(dst.ckpt._baseline - 1.25) < 1e-6
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
    src.checkpoint_store = FileCheckpointStore()
    src.ckpt = _scaffold()
    src.train_steps_count = 3
    src._ckpt_update_residual = 0.0
    src.eval_snapshot = None
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))
    src.simba = True
    src.reward_normalizer = None
    src.obs_rms = RunningMeanStd(shapes={"obs": (2,)}, dtype=np.float64)
    src.obs_rms.update({"obs": np.ones((4, 2), np.float64)})
    src.action_obs_rms = RunningMeanStd.from_state(src.obs_rms.to_state())
    src.checkpoint_obs_rms = None  # exercises the None branch

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = DDPG.__new__(DDPG)
        dst.checkpoint_store = FileCheckpointStore()
        dst.simba = True
        dst.reward_normalizer = None
        dst.ckpt = _scaffold()
        dst.load_params(d)

    assert dst.obs_rms is not None
    assert dst.action_obs_rms is not None
    assert dst.checkpoint_obs_rms is None
    assert dst.eval_snapshot is None
    _assert_tree_equal(src.obs_rms.to_state(), dst.obs_rms.to_state())


def test_dpg_reward_normalizer_statistics_round_trip_without_partial_returns():
    fields = ALGO_FIELDS[DDPG]
    src = DDPG.__new__(DDPG)
    src.checkpoint_store = FileCheckpointStore()
    src.ckpt = _scaffold()
    src.train_steps_count = 3
    src._ckpt_update_residual = 0.0
    src.eval_snapshot = None
    src.simba = False
    src.reward_normalizer = RewardNormalizer(worker_size=2, gamma=0.9)
    src.reward_normalizer.record(
        rewards=np.array([1.0, 3.0]),
        dones=np.array([False, False]),
    )
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = DDPG.__new__(DDPG)
        dst.checkpoint_store = FileCheckpointStore()
        dst.ckpt = _scaffold()
        dst.simba = False
        dst.reward_normalizer = RewardNormalizer(worker_size=3, gamma=0.9)
        dst.load_params(d)

    _assert_tree_equal(src.reward_normalizer.to_state(), dst.reward_normalizer.to_state())
    np.testing.assert_array_equal(dst.reward_normalizer.discounted_returns, np.zeros(3))


def test_qnet_reward_normalizer_statistics_round_trip_without_partial_returns():
    src = Q_Network_Family.__new__(Q_Network_Family)
    src.checkpoint_store = FileCheckpointStore()
    src.params = _tree(1.0)
    src.reward_normalizer = RewardNormalizer(worker_size=2, gamma=0.99)
    src.reward_normalizer.record(
        rewards=np.array([2.0, 4.0]),
        dones=np.array([False, False]),
    )

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = Q_Network_Family.__new__(Q_Network_Family)
        dst.checkpoint_store = FileCheckpointStore()
        dst.reward_normalizer = RewardNormalizer(worker_size=3, gamma=0.99)
        dst.load_params(d)

    _assert_tree_equal(src.params, dst.params)
    assert dst.target_params is dst.params
    _assert_tree_equal(src.reward_normalizer.to_state(), dst.reward_normalizer.to_state())
    np.testing.assert_array_equal(dst.reward_normalizer.discounted_returns, np.zeros(3))


def test_qnet_legacy_checkpoint_clears_partial_discounted_returns():
    src = Q_Network_Family.__new__(Q_Network_Family)
    src.checkpoint_store = FileCheckpointStore()
    src.reward_normalizer = None
    src.params = _tree(1.0)

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = Q_Network_Family.__new__(Q_Network_Family)
        dst.checkpoint_store = FileCheckpointStore()
        dst.reward_normalizer = RewardNormalizer(worker_size=2, gamma=0.99)
        dst.reward_normalizer.record(
            rewards=np.array([2.0, 4.0]),
            dones=np.array([False, False]),
        )
        dst.load_params(d)

    np.testing.assert_array_equal(dst.reward_normalizer.discounted_returns, np.zeros(2))


def test_dpg_checkpoint_without_reward_stats_clears_partial_discounted_returns():
    fields = ALGO_FIELDS[DDPG]
    src = DDPG.__new__(DDPG)
    src.checkpoint_store = FileCheckpointStore()
    src.ckpt = _scaffold()
    src.train_steps_count = 3
    src._ckpt_update_residual = 0.0
    src.eval_snapshot = None
    src.simba = False
    src.reward_normalizer = None
    for i, name in enumerate(fields):
        setattr(src, name, _field_value(name, i))

    with tempfile.TemporaryDirectory() as d:
        src.save_params(d)
        dst = DDPG.__new__(DDPG)
        dst.checkpoint_store = FileCheckpointStore()
        dst.ckpt = _scaffold()
        dst.simba = False
        dst.reward_normalizer = RewardNormalizer(worker_size=2, gamma=0.99)
        dst.reward_normalizer.record(
            rewards=np.array([2.0, 4.0]),
            dones=np.array([False, False]),
        )
        dst.load_params(d)

    np.testing.assert_array_equal(dst.reward_normalizer.discounted_returns, np.zeros(2))


def test_base_contract_is_abstract():
    from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family

    base = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    with pytest.raises(NotImplementedError):
        base.checkpoint_params()
    with pytest.raises(NotImplementedError):
        base.load_checkpoint_params(object())
