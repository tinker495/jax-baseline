"""Regression tests for the on-policy Actor-Critic ``learn`` keyword contract.

The synchronous PG family (A2C / PPO / TPPO / SPO) all inherit
``Actor_Critic_Policy_Gradient_Family.learn``, which expects ``experiment_name``
before ``run_name``. The CLI (``experiments/cli/pg.py``) calls
``agent.learn(steps, experiment_name=...)`` by keyword, so every subclass must
expose those two parameters in the same positional order as the base to avoid
silent positional drift (Hyrum's-law breakage for positional callers).

Import-only: no Ray / env construction required.
"""

import inspect

import pytest

from jax_baselines.A2C.a2c import A2C
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.PPO.ppo import PPO
from jax_baselines.SPO.spo import SPO
from jax_baselines.TPPO.tppo import TPPO


@pytest.mark.parametrize("cls", [A2C, PPO, TPPO, SPO])
def test_learn_exposes_experiment_then_run_name(cls):
    params = list(inspect.signature(cls.learn).parameters)
    assert "experiment_name" in params
    assert "run_name" in params
    # experiment_name must precede run_name, matching the family base contract.
    assert params.index("experiment_name") < params.index("run_name")


@pytest.mark.parametrize("cls", [A2C, PPO, TPPO, SPO])
def test_learn_param_order_matches_base(cls):
    base_params = list(inspect.signature(Actor_Critic_Policy_Gradient_Family.learn).parameters)
    sub_params = list(inspect.signature(cls.learn).parameters)
    base_pos = (base_params.index("experiment_name"), base_params.index("run_name"))
    sub_pos = (sub_params.index("experiment_name"), sub_params.index("run_name"))
    assert base_pos == sub_pos
