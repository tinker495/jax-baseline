"""Ray-free guard: APE-X leaves keep their per-algorithm run_name default after the
learn() wrappers were consolidated onto the shared base via the `_run_name` attribute."""
import inspect

import pytest

from jax_baselines.C51.apex_c51 import APE_X_C51
from jax_baselines.DDPG.apex_ddpg import APE_X_DDPG
from jax_baselines.DQN.apex_dqn import APE_X_DQN
from jax_baselines.IQN.apex_iqn import APE_X_IQN
from jax_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN
from jax_baselines.TD3.apex_td3 import APE_X_TD3

EXPECTED = {
    APE_X_DQN: "Ape_X_DQN",
    APE_X_QRDQN: "Ape_X_QRDQN",
    APE_X_IQN: "Ape_X_IQN",
    APE_X_C51: "Ape_X_C51",
    APE_X_DDPG: "Ape_X_DDPG",
    APE_X_TD3: "Ape_X_TD3",
}


@pytest.mark.parametrize("cls,name", list(EXPECTED.items()))
def test_apex_run_name_default(cls, name):
    assert cls._run_name == name


@pytest.mark.parametrize("cls", list(EXPECTED))
def test_apex_leaf_inherits_learn(cls):
    # learn() now lives on the shared base, not the leaf.
    assert "learn" not in cls.__dict__
    assert "run_name" in inspect.signature(cls.learn).parameters
