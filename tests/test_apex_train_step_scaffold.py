"""Ray-free guards for the APE-X ``train_step`` scaffolding extraction.

The identical per-leaf ``train_step`` minibatch loop was hoisted onto the two APE-X
bases (``Ape_X_Family`` and ``Ape_X_Deteministic_Policy_Gradient_Family``); the one
divergent line -- the ``_train_step(...)`` call, whose argument shape differs per
algorithm -- stays in a per-class ``_invoke_train_step(self, steps, data)`` hook.

These tests pin (a) the shared base orchestration, (b) each leaf's exact forwarding
into ``_train_step``, and (c) the bundled fix that ``train_steps_count`` is now
initialized on the APE-X bases (it was incremented every ``train_step`` but never
initialized -- an ``AttributeError`` on the first distributed gradient step). No Ray,
no model build, no replay buffer.
"""

import inspect
from types import SimpleNamespace

import pytest

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.C51.apex_c51 import APE_X_C51
from jax_baselines.DDPG.apex_ddpg import APE_X_DDPG
from jax_baselines.DQN.apex_dqn import APE_X_DQN
from jax_baselines.IQN.apex_iqn import APE_X_IQN
from jax_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN
from jax_baselines.TD3.apex_td3 import APE_X_TD3

APEX_ALL = [APE_X_DQN, APE_X_QRDQN, APE_X_IQN, APE_X_C51, APE_X_DDPG, APE_X_TD3]
BASES = [Ape_X_Family, Ape_X_Deteministic_Policy_Gradient_Family]


class _RecordingReplay:
    def __init__(self):
        self.update_calls = []

    def sample(self, batch_size, beta):
        return {"indexes": [0, 1], "obses": "OBS"}

    def update_priorities(self, indexes, priorities):
        self.update_calls.append((indexes, priorities))


def _base_fake(dpg):
    invoked = []
    logged = []
    return (
        SimpleNamespace(
            train_steps_count=0,
            batch_size=4,
            prioritized_replay_beta0=0.4,
            log_interval=1,
            replay_buffer=_RecordingReplay(),
            logger_server=SimpleNamespace(log_trainer=lambda steps, d: logged.append((steps, d))),
            _invoke_train_step=lambda steps, data: (
                invoked.append((steps, data))
                or (
                    ("P", "C", "TP", "TC", "POPT", "COPT", 0.5, 1.5, [0.1, 0.2])
                    if dpg
                    else ("P", "TP", "OPT", 0.5, 1.5, [0.1, 0.2])
                )
            ),
        ),
        invoked,
        logged,
    )


@pytest.mark.parametrize("base_cls", BASES)
def test_base_train_step_orchestration(base_cls):
    obj, invoked, logged = _base_fake(base_cls is Ape_X_Deteministic_Policy_Gradient_Family)
    loss = base_cls.train_step(obj, steps=10, gradient_steps=3)

    assert len(invoked) == 3  # one hook call per gradient step
    assert all(s == 10 for s, _ in invoked)
    assert obj.train_steps_count == 3  # counter advanced from its initialized 0
    if base_cls is Ape_X_Deteministic_Policy_Gradient_Family:
        assert (obj.policy_params, obj.critic_params) == ("P", "C")
        assert (obj.target_policy_params, obj.target_critic_params) == ("TP", "TC")
        assert (obj.opt_policy_state, obj.opt_critic_state) == ("POPT", "COPT")
    else:
        assert (obj.params, obj.target_params, obj.opt_state) == ("P", "TP", "OPT")
    assert obj.replay_buffer.update_calls == [([0, 1], [0.1, 0.2])] * 3
    assert loss == 0.5  # last loss returned
    assert logged == [(10, {"loss/qloss": 0.5, "loss/targets": 1.5})]


def _leaf_fake():
    captured = {}

    def fake_train_step(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ("P", "TP", "OPT", 0.0, 0.0, [])

    return (
        SimpleNamespace(
            params="P0",
            target_params="TP0",
            opt_state="O0",
            policy_params="P0",
            critic_params="C0",
            target_policy_params="TP0",
            target_critic_params="TC0",
            opt_policy_state="PO0",
            opt_critic_state="CO0",
            key_seq=iter(["K0", "K1", "K2"]),
            param_noise=False,
            _train_step=fake_train_step,
            _compiled_train_step=fake_train_step,
        ),
        captured,
    )


@pytest.mark.parametrize("cls", APEX_ALL)
def test_leaf_invoke_forwards_params_and_data(cls):
    obj, captured = _leaf_fake()
    cls._invoke_train_step(obj, steps=7, data={"obses": "X"})
    if cls in (APE_X_DDPG, APE_X_TD3):
        assert captured["args"][:6] == ("P0", "C0", "TP0", "TC0", "PO0", "CO0")
    else:
        assert captured["args"][:3] == ("P0", "TP0", "O0")
    assert captured["kwargs"] == {"obses": "X"}


def test_dqn_family_invoke_passes_steps_then_key():
    for cls in (APE_X_DQN, APE_X_QRDQN, APE_X_C51):
        obj, captured = _leaf_fake()
        cls._invoke_train_step(obj, steps=7, data={})
        assert captured["args"] == ("P0", "TP0", "O0", 7, "K0")


def test_ddpg_invoke_omits_steps_and_passes_scan_key():
    obj, captured = _leaf_fake()
    APE_X_DDPG._invoke_train_step(obj, steps=7, data={})
    assert captured["args"] == ("P0", "C0", "TP0", "TC0", "PO0", "CO0", "K0")


def test_td3_invoke_passes_key_then_steps():
    obj, captured = _leaf_fake()
    APE_X_TD3._invoke_train_step(obj, steps=7, data={})
    assert captured["args"] == ("P0", "C0", "TP0", "TC0", "PO0", "CO0", "K0", 7)


def test_iqn_invoke_gates_key_on_param_noise():
    obj, captured = _leaf_fake()
    obj.param_noise = False
    APE_X_IQN._invoke_train_step(obj, steps=7, data={})
    assert captured["args"] == ("P0", "TP0", "O0", 7, None)

    obj2, captured2 = _leaf_fake()
    obj2.param_noise = True
    APE_X_IQN._invoke_train_step(obj2, steps=7, data={})
    assert captured2["args"] == ("P0", "TP0", "O0", 7, "K0")


@pytest.mark.parametrize("cls", APEX_ALL)
def test_leaf_inherits_train_step_and_owns_hook(cls):
    assert cls.train_step in (
        Ape_X_Family.train_step,
        Ape_X_Deteministic_Policy_Gradient_Family.train_step,
    )
    assert cls._invoke_train_step.__qualname__ == f"{cls.__name__}._invoke_train_step"


@pytest.mark.parametrize("base_cls", BASES)
def test_base_initializes_train_steps_count(base_cls):
    # bundled latent fix: the counter incremented in train_step is initialized here.
    assert "self.train_steps_count = 0" in inspect.getsource(base_cls.__init__)
