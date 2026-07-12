"""Regression coverage for family-owned ``learn`` entrypoints.

Algorithm leaves only provide their historical run-name defaults.  The four
family bases own the entrypoint and resolve those defaults at call time.
"""

from __future__ import annotations

import importlib

import pytest

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.A2C.impala import IMPALA
from jax_baselines.BBF.bbf import BBF
from jax_baselines.BBF.hl_gauss_bbf import HL_GAUSS_BBF
from jax_baselines.C51.c51 import C51
from jax_baselines.C51.hl_gauss_c51 import HL_GAUSS_C51
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.DQN.dqn import DQN
from jax_baselines.FQF.fqf import FQF
from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.IQN.iqn import IQN
from jax_baselines.PPO.impala_ppo import IMPALA_PPO
from jax_baselines.PPO.ppo import PPO
from jax_baselines.QRDQN.qrdqn import QRDQN
from jax_baselines.SAC.sac import SAC
from jax_baselines.SPO.impala_spo import IMPALA_SPO
from jax_baselines.SPO.spo import SPO
from jax_baselines.SPR.hl_gauss_spr import HL_GAUSS_SPR
from jax_baselines.SPR.spr import SPR
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TPPO.impala_tppo import IMPALA_TPPO
from jax_baselines.TPPO.tppo import TPPO
from jax_baselines.TQC.tqc import TQC

_LOCAL_LEAVES = (
    (DQN, "DQN", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (C51, "C51", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (HL_GAUSS_C51, "HL_GAUSS_C51", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (FQF, "FQF", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (IQN, "IQN", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (QRDQN, "QRDQN", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (SPR, "SPR", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (HL_GAUSS_SPR, "HL_GAUSS_SPR", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (BBF, "BBF", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (HL_GAUSS_BBF, "HL_GAUSS_BBF", Q_Network_Family, "jax_baselines.DQN.base_class"),
    (
        DDPG,
        "DDPG",
        Deteministic_Policy_Gradient_Family,
        "jax_baselines.DDPG.base_class",
    ),
    (TD3, "TD3", Deteministic_Policy_Gradient_Family, "jax_baselines.DDPG.base_class"),
    (SAC, "SAC", Deteministic_Policy_Gradient_Family, "jax_baselines.DDPG.base_class"),
    (TQC, "TQC", Deteministic_Policy_Gradient_Family, "jax_baselines.DDPG.base_class"),
    (
        CrossQ,
        "CrossQ",
        Deteministic_Policy_Gradient_Family,
        "jax_baselines.DDPG.base_class",
    ),
    (TD7, "TD7", Deteministic_Policy_Gradient_Family, "jax_baselines.DDPG.base_class"),
    (PPO, "PPO", Actor_Critic_Policy_Gradient_Family, "jax_baselines.A2C.base_class"),
    (SPO, "SPO", Actor_Critic_Policy_Gradient_Family, "jax_baselines.A2C.base_class"),
    (TPPO, "TPPO", Actor_Critic_Policy_Gradient_Family, "jax_baselines.A2C.base_class"),
)

_IMPALA_LEAVES = (
    (IMPALA, "IMPALA_AC"),
    (IMPALA_PPO, "IMPALA_PPO"),
    (IMPALA_SPO, "IMPALA_SPO"),
    (IMPALA_TPPO, "IMPALA_TPPO"),
)

_ALL_LEAVES = tuple((cls, name, base) for cls, name, base, _module in _LOCAL_LEAVES) + tuple(
    (cls, name, IMPALA_Family) for cls, name in _IMPALA_LEAVES
)


@pytest.mark.parametrize(
    "cls,expected_name,base",
    _ALL_LEAVES,
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_leaf_inherits_family_learn(cls, expected_name, base):
    assert cls._run_name == expected_name
    assert "learn" not in cls.__dict__
    assert cls.learn is base.learn


@pytest.mark.parametrize(
    "cls,expected_name,_base,module_name",
    _LOCAL_LEAVES,
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_local_family_resolves_leaf_defaults_at_call_time(
    monkeypatch, cls, expected_name, _base, module_name
):
    calls = []

    class _Session:
        def run(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "trained"

    monkeypatch.setattr(importlib.import_module(module_name), "TrainingSession", _Session)
    agent = cls.__new__(cls)

    assert agent.learn(123) == "trained"
    args, kwargs = calls.pop()
    assert args[4:6] == (expected_name, expected_name)
    assert kwargs == {
        "logger_factory": None,
        "progress_factory": None,
        "record_test_fn": None,
    }

    agent.learn(123, experiment_name="", run_name="")
    args, _kwargs = calls.pop()
    assert args[4:6] == ("", "")


@pytest.mark.parametrize(
    "cls,expected_name",
    _IMPALA_LEAVES,
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_impala_family_resolves_leaf_defaults_at_call_time(cls, expected_name):
    logger_calls = []
    progress_calls = []
    saved_paths = []

    class _LoggerServer:
        def get_log_dir(self):
            return "/tmp/impala"

        def last_update(self):
            pass

        def close(self):
            pass

    class _Runtime:
        def create_logger_server(self, log_dir, run_name, experiment_name, logger_factory):
            logger_calls.append((log_dir, run_name, experiment_name, logger_factory))
            return _LoggerServer()

        def shutdown(self):
            pass

    def progress_factory(total, *, miniters):
        progress_calls.append((total, miniters))
        return ()

    agent = cls.__new__(cls)
    agent.runtime = _Runtime()
    agent.log_dir = None
    agent.env_type = "none"
    agent.save_params = saved_paths.append
    expected_logger_name = expected_name
    if cls is IMPALA_TPPO:
        agent.mu_ratio = 0.5
        expected_logger_name += "(0.50)"

    agent.learn(123, progress_factory=progress_factory)
    assert progress_calls.pop() == (123, 10)
    assert logger_calls.pop()[1:3] == (expected_logger_name, "experiment")
    assert saved_paths.pop() == "/tmp/impala"

    if cls is IMPALA_TPPO:
        agent.mu_ratio = 0.0
    agent.learn(123, log_interval=0, run_name="", progress_factory=progress_factory)
    assert progress_calls.pop() == (123, 0)
    assert logger_calls.pop()[1] == ""


def test_ddpg_prepare_run_keeps_exploration_initialization():
    agent = DDPG.__new__(DDPG)
    agent.exploration_fraction = 0.5
    agent.exploration_initial_eps = 1.0
    agent.exploration_final_eps = 0.1

    agent.prepare_run(100)

    assert agent.epsilon == 1.0
    assert agent.exploration.value(0) == pytest.approx(1.0)
    assert agent.exploration.value(50) == pytest.approx(0.1)


def _set_q_run_name_flags(agent):
    agent.munchausen = True
    agent.param_noise = False
    agent.dueling_model = False
    agent.double_q = False
    agent.n_step_method = False
    agent.prioritized_replay = True


@pytest.mark.parametrize(
    "cls,attrs,expected",
    (
        (FQF, {"n_support": 32}, "M-FQF(32)+PER"),
        (QRDQN, {"n_support": 32}, "M-QRDQN(32)+PER"),
        (IQN, {"n_support": 32, "risk_avoid": False, "CVaR": 1.0}, "M-IQN(32)+PER"),
        (
            IQN,
            {"n_support": 32, "risk_avoid": True, "CVaR": 0.25},
            "M-IQN(32)_CVaR(0.25)+PER",
        ),
    ),
)
def test_quantile_run_names_keep_leaf_suffix_before_family_decoration(cls, attrs, expected):
    agent = cls.__new__(cls)
    _set_q_run_name_flags(agent)
    for name, value in attrs.items():
        setattr(agent, name, value)

    assert agent.run_name_update(cls._run_name) == expected


@pytest.mark.parametrize(
    "mixture_type,expected",
    (
        ("truncated", "3Step_Simba_TQC(25)_truncated(4)+PER"),
        ("min", "3Step_Simba_TQC(25)_min+PER"),
    ),
)
def test_tqc_run_name_keeps_quantile_suffix_before_family_decoration(mixture_type, expected):
    agent = TQC.__new__(TQC)
    agent.n_support = 25
    agent.mixture_type = mixture_type
    agent.quantile_drop = 4
    agent.simba_v2 = False
    agent.simba = True
    agent.n_step_method = True
    agent.n_step = 3
    agent.prioritized_replay = True

    assert agent.run_name_update(agent._run_name) == expected


def test_impala_tppo_run_name_keeps_mu_ratio_suffix():
    agent = IMPALA_TPPO.__new__(IMPALA_TPPO)
    agent.mu_ratio = 0.5

    assert agent.run_name_update(agent._run_name) == "IMPALA_TPPO(0.50)"
