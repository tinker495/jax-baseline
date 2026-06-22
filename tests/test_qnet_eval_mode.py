"""Regression tests: IQN/FQF ``actions()`` must honor ``eval_mode``.

IQN and FQF override ``actions()`` because their ``_get_actions`` signatures differ
from the Q-family base (``DQN/base_class.py``). The overrides must still apply the
base contract: during evaluation with checkpointing enabled, behavior selects the
frozen ``checkpoint_params`` snapshot instead of the live ``params``. A prior version
ignored ``eval_mode`` entirely and always used the live params.

The methods are exercised against a lightweight stub so no JAX model is built;
``epsilon=0.0`` forces the greedy branch deterministically so the param selection
is observable.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.FQF.fqf import FQF
from jax_baselines.IQN.iqn import IQN

BEHAVIOR_PARAMS = "behavior-params"
CHECKPOINT_PARAMS = "checkpoint-params"


class _Ckpt:
    def __init__(self, enabled):
        self.enabled = enabled


class _ActorStub:
    """Minimal stand-in exposing exactly what IQN/FQF ``actions()`` read."""

    def __init__(self, use_checkpointing=True, ckpt_enabled=True, param_noise=False):
        self.params = BEHAVIOR_PARAMS
        self.fqf_params = "fqf-params"
        self.checkpoint_params = CHECKPOINT_PARAMS
        self.use_checkpointing = use_checkpointing
        self.ckpt = _Ckpt(ckpt_enabled)
        self.param_noise = param_noise
        self.key_seq = iter(range(1000))
        self.action_size = [3]
        self.worker_size = 1
        self.used_params = []

    def get_behavior_params(self):
        return self.params

    def _get_actions(self, params, *rest):
        self.used_params.append(params)
        return np.zeros((1, 1))


def test_iqn_actions_uses_checkpoint_params_in_eval_mode():
    stub = _ActorStub()
    IQN.actions(stub, np.zeros((1, 4)), 0.0, eval_mode=True)
    assert stub.used_params == [CHECKPOINT_PARAMS]


def test_iqn_actions_uses_behavior_params_when_not_eval():
    stub = _ActorStub()
    IQN.actions(stub, np.zeros((1, 4)), 0.0, eval_mode=False)
    assert stub.used_params == [BEHAVIOR_PARAMS]


def test_iqn_actions_ignores_checkpoint_when_checkpointing_disabled():
    stub = _ActorStub(use_checkpointing=False)
    IQN.actions(stub, np.zeros((1, 4)), 0.0, eval_mode=True)
    assert stub.used_params == [BEHAVIOR_PARAMS]


def test_iqn_get_actions_uses_observation_batch_for_tau_not_worker_size():
    stub = IQN.__new__(IQN)
    stub.worker_size = 32
    stub.n_support = 8
    stub.CVaR = 1.0
    stub.action_size = [3]
    seen = {}

    def get_q(params, obses, tau, key=None):
        del params, obses, key
        seen["tau_shape"] = tuple(tau.shape)
        return jnp.zeros((tau.shape[0], stub.action_size[0], tau.shape[1]))

    stub.get_q = get_q

    actions = IQN._get_actions(
        stub,
        BEHAVIOR_PARAMS,
        [np.zeros((1, 4), dtype=np.float32)],
        jax.random.PRNGKey(0),
    )

    assert seen["tau_shape"] == (1, stub.n_support)
    assert np.asarray(actions).shape == (1, 1)


def test_fqf_actions_uses_checkpoint_params_in_eval_mode():
    stub = _ActorStub()
    FQF.actions(stub, np.zeros((1, 4)), 0.0, eval_mode=True)
    assert stub.used_params == [CHECKPOINT_PARAMS]


def test_fqf_actions_uses_behavior_params_when_not_eval():
    stub = _ActorStub()
    FQF.actions(stub, np.zeros((1, 4)), 0.0, eval_mode=False)
    assert stub.used_params == [BEHAVIOR_PARAMS]
