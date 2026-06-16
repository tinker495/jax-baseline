"""W2: PPO/SPO and IMPALA_PPO/IMPALA_SPO each share identical preprocessing +
minibatch/epoch optimization, extracted into a shared surrogate base.

Each pair must resolve its shared methods to the single base implementation (no
divergent copies), while keeping its own distinct ``__init__``/``_loss_*``/``learn``.
"""

import pytest

from jax_baselines.A2C.surrogate_base import SurrogatePolicyGradient
from jax_baselines.IMPALA.surrogate_base import SurrogateIMPALA
from jax_baselines.PPO.impala_ppo import IMPALA_PPO
from jax_baselines.PPO.ppo import PPO
from jax_baselines.SPO.impala_spo import IMPALA_SPO
from jax_baselines.SPO.spo import SPO

# (base, ppo_variant, spo_variant, shared_method_names)
_PAIRS = [
    (
        SurrogatePolicyGradient,
        PPO,
        SPO,
        ("setup_model", "train_step", "_preprocess", "_train_step"),
    ),
    (
        SurrogateIMPALA,
        IMPALA_PPO,
        IMPALA_SPO,
        ("setup_model", "train_step", "preprocess", "_train_step"),
    ),
]
_ALGO_SPECIFIC = ("__init__", "_loss_discrete", "_loss_continuous", "learn")


@pytest.mark.parametrize("base, ppo_cls, spo_cls, shared", _PAIRS)
def test_pair_inherits_shared_base(base, ppo_cls, spo_cls, shared):
    assert issubclass(ppo_cls, base)
    assert issubclass(spo_cls, base)


@pytest.mark.parametrize("base, ppo_cls, spo_cls, shared", _PAIRS)
def test_shared_methods_are_single_sourced(base, ppo_cls, spo_cls, shared):
    for name in shared:
        ref = getattr(base, name)
        assert getattr(ppo_cls, name) is ref, f"{ppo_cls.__name__}.{name} diverged from base"
        assert getattr(spo_cls, name) is ref, f"{spo_cls.__name__}.{name} diverged from base"
        # Not copied back into the subclass __dict__.
        assert name not in ppo_cls.__dict__
        assert name not in spo_cls.__dict__


@pytest.mark.parametrize("base, ppo_cls, spo_cls, shared", _PAIRS)
def test_loss_and_init_stay_algorithm_specific(base, ppo_cls, spo_cls, shared):
    for name in _ALGO_SPECIFIC:
        assert name in ppo_cls.__dict__, f"{ppo_cls.__name__} must define its own {name}"
        assert name in spo_cls.__dict__, f"{spo_cls.__name__} must define its own {name}"
        assert ppo_cls.__dict__[name] is not spo_cls.__dict__[name]
