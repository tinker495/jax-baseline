import pytest

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.BBF.hl_gauss_bbf import HL_GAUSS_BBF
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.FQF.fqf import FQF
from jax_baselines.optim import require_optimizer_factory


class FakeOptimizer:
    def __init__(self, calls):
        self.calls = calls

    def init(self, params):
        self.calls.append({"init_params": params})
        return {"state": "reset"}


def test_require_optimizer_factory_documents_experiment_side_contract():
    with pytest.raises(ValueError, match="resolve optimizer names/defaults in experiments"):
        require_optimizer_factory(None)


def test_a2c_make_optimizer_uses_injected_factory():
    calls = []

    def factory(learning_rate):
        calls.append({"learning_rate": learning_rate})
        return FakeOptimizer(calls)

    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.optimizer_factory = factory

    optimizer = agent._make_optimizer(0.00025)

    assert isinstance(optimizer, FakeOptimizer)
    assert calls == [{"learning_rate": 0.00025}]


def test_a2c_prepare_run_rebuilds_optimizer_with_linear_lr_schedule_through_factory():
    calls = []

    def factory(learning_rate):
        calls.append({"learning_rate": learning_rate})
        return FakeOptimizer(calls)

    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.optimizer_factory = factory
    agent.learning_rate = 0.00025
    agent.lr_annealing = True
    agent.batch_size = 10
    agent.worker_size = 2
    agent.minibatch_size = 5
    agent.epoch_num = 3
    agent.params = {"w": 1.0}
    agent.opt_state = {"state": "old"}

    agent.prepare_run(100)

    schedule = calls[0]["learning_rate"]
    assert agent._lr_annealing_transition_steps(100) == 60
    assert callable(schedule)
    assert float(schedule(0)) == pytest.approx(0.00025)
    assert float(schedule(30)) == pytest.approx(0.000125)
    assert float(schedule(60)) == pytest.approx(0.0)
    assert calls[1] == {"init_params": {"w": 1.0}}
    assert agent.opt_state == {"state": "reset"}


def test_q_network_constructor_uses_injected_optimizer_factory(monkeypatch):
    calls = []
    optimizer = FakeOptimizer(calls)

    def factory(learning_rate):
        calls.append({"learning_rate": learning_rate})
        return optimizer

    monkeypatch.setattr(Q_Network_Family, "get_env_setup", lambda self: None)
    monkeypatch.setattr(Q_Network_Family, "get_memory_setup", lambda self: None)

    agent = Q_Network_Family(
        env_builder=lambda: None,
        model_builder_maker=lambda *args, **kwargs: None,
        optimizer_factory=factory,
        learning_rate=0.125,
        _init_setup_model=False,
    )

    assert agent.optimizer is optimizer
    assert calls == [{"learning_rate": 0.125}]


def test_fqf_second_optimizer_uses_injected_factory():
    calls = []
    optimizer = FakeOptimizer(calls)

    def factory(learning_rate):
        calls.append({"learning_rate": learning_rate})
        return optimizer

    agent = FQF.__new__(FQF)
    agent.fqf_optimizer_factory = factory
    agent.learning_rate = 0.2
    agent.fqf_factor = 0.01

    assert agent._make_fqf_optimizer() is optimizer
    assert calls == [{"learning_rate": pytest.approx(0.002)}]


def test_hl_gauss_bbf_setup_model_keeps_optimizer_policy_on_factory_seam():
    names = HL_GAUSS_BBF.setup_model.__code__.co_names

    assert "adamw" not in names
    assert "_make_optimizer" in names
