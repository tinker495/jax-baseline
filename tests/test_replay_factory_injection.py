from pathlib import Path
from types import SimpleNamespace

import pytest

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.replay_protocol import (
    LocalReplayNeed,
    PriorityNeed,
    SelfPredictionReplayNeed,
    make_worker_local_replay_buffer,
)
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DQN.base_class import Q_Network_Family


class FakeLocalReplayFactory:
    def __init__(self, buffer):
        self.buffer = buffer
        self.calls = []

    def __call__(self, need):
        self.calls.append(need)
        return self.buffer


class FakeReplayFactory:
    def __init__(self, buffer):
        self.buffer = buffer
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.buffer


class FakeWorkerReplayFactory:
    def __init__(self, buffer):
        self.buffer = buffer
        self.calls = []

    def __call__(self, local_size, *, env_dict, n_s=None):
        self.calls.append({"local_size": local_size, "env_dict": env_dict, "n_s": n_s})
        return self.buffer


def test_q_network_family_uses_injected_replay_factory():
    agent = Q_Network_Family.__new__(Q_Network_Family)
    fake_buffer = object()
    factory = FakeLocalReplayFactory(fake_buffer)
    agent.replay_factory = factory
    agent.buffer_size = 123
    agent.observation_space = [[4]]
    agent.worker_size = 2
    agent.n_step = 3
    agent.n_step_method = True
    agent.gamma = 0.9
    agent.prioritized_replay = True
    agent.prioritized_replay_alpha = 0.7
    agent.prioritized_replay_eps = 0.01
    agent.compress_memory = True

    agent.get_memory_setup()

    assert agent.replay_buffer is fake_buffer
    assert factory.calls == [
        LocalReplayNeed(
            buffer_size=123,
            observation_space=[[4]],
            action_shape_or_n=1,
            worker_size=2,
            n_step=3,
            gamma=0.9,
            priority=PriorityNeed(alpha=0.7, eps=0.01),
            compress_observations=True,
        )
    ]


def test_dpg_family_uses_injected_replay_factory():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    fake_buffer = object()
    factory = FakeLocalReplayFactory(fake_buffer)
    agent.replay_factory = factory
    agent.buffer_size = 456
    agent.observation_space = [[8]]
    agent.action_size = [2]
    agent.worker_size = 4
    agent.n_step = 2
    agent.n_step_method = True
    agent.gamma = 0.95
    agent.prioritized_replay = False
    agent.prioritized_replay_alpha = 0.6
    agent.prioritized_replay_eps = 0.001

    agent.get_memory_setup()

    assert agent.replay_buffer is fake_buffer
    assert factory.calls == [
        LocalReplayNeed(
            buffer_size=456,
            observation_space=[[8]],
            action_shape_or_n=[2],
            worker_size=4,
            n_step=2,
            gamma=0.95,
            priority=None,
        )
    ]


@pytest.mark.parametrize(
    "family_cls,action_shape_or_n",
    [
        (Ape_X_Family, 1),
        (Ape_X_Deteministic_Policy_Gradient_Family, [3]),
    ],
)
def test_apex_families_use_injected_shared_and_worker_factories(family_cls, action_shape_or_n):
    agent = family_cls.__new__(family_cls)
    fake_buffer = object()
    multi_factory = FakeReplayFactory(fake_buffer)
    agent.multi_replay_factory = multi_factory
    agent.worker_replay_factory = FakeWorkerReplayFactory(object())
    agent.buffer_size = 789
    agent.observation_space = [[5]]
    agent.action_size = [3]
    agent.prioritized_replay_alpha = 0.8
    agent.prioritized_replay_eps = 0.02
    agent.n_step = 4
    agent.gamma = 0.91
    agent.runtime = SimpleNamespace(replay_manager=lambda: "manager")
    agent.compress_memory = True

    agent.get_memory_setup()

    assert agent.replay_buffer is fake_buffer
    assert multi_factory.calls == [
        {
            "buffer_size": 789,
            "observation_space": [[5]],
            "alpha": 0.8,
            "action_shape_or_n": action_shape_or_n,
            "n_step": 4,
            "gamma": 0.91,
            "manager": "manager",
            "compress_memory": True,
            "eps": 0.02,
        }
    ]


def test_spr_uses_injected_transition_replay_factory():
    from jax_baselines.SPR.spr import SPR

    agent = SPR.__new__(SPR)
    fake_buffer = object()
    factory = FakeLocalReplayFactory(fake_buffer)
    agent.replay_factory = factory
    agent.buffer_size = 321
    agent.observation_space = [[84, 84, 4]]
    agent.worker_size = 1
    agent.n_step = 3
    agent.n_step_method = True
    agent.prediction_depth = 5
    agent.gamma = 0.99
    agent.prioritized_replay = True
    agent.prioritized_replay_alpha = 0.6
    agent.prioritized_replay_eps = 0.01
    agent.compress_memory = False

    agent.get_memory_setup()

    assert agent.replay_buffer is fake_buffer
    assert factory.calls == [
        SelfPredictionReplayNeed(
            buffer_size=321,
            observation_space=[[84, 84, 4]],
            action_shape_or_n=1,
            worker_size=1,
            n_step=3,
            gamma=0.99,
            priority=PriorityNeed(alpha=0.6, eps=0.01),
            compress_observations=False,
            prediction_depth=5,
        )
    ]


def test_transition_replay_buffers_live_in_replay_adapter():
    assert not Path("jax_baselines/SPR/efficent_buffer.py").exists()
    source = Path("jax_baselines/SPR/spr.py").read_text()
    assert "TransitionReplayBuffer" not in source
    assert "require_replay_factory" in source
    assert Path("replay_memory/transition_buffers.py").exists()


def test_replay_factory_builds_self_prediction_buffer_from_need():
    from replay_memory.replay_factory import make_replay_buffer
    from replay_memory.transition_buffers import PrioritizedTransitionReplayBuffer

    buf = make_replay_buffer(
        SelfPredictionReplayNeed(
            buffer_size=321,
            observation_space=[[84, 84, 4]],
            action_shape_or_n=1,
            n_step=3,
            gamma=0.99,
            priority=PriorityNeed(alpha=0.6, eps=0.01),
            prediction_depth=5,
        )
    )

    assert isinstance(buf, PrioritizedTransitionReplayBuffer)
    assert buf.prediction_depth == 5


@pytest.mark.parametrize(
    "need,match",
    [
        (
            SelfPredictionReplayNeed(
                buffer_size=321,
                observation_space=[[84, 84, 4]],
                action_shape_or_n=1,
                compress_observations=True,
                prediction_depth=5,
            ),
            "compress_observations=True",
        ),
        (
            SelfPredictionReplayNeed(
                buffer_size=321,
                observation_space=[[84, 84, 4]],
                action_shape_or_n=1,
                worker_size=2,
                prediction_depth=5,
            ),
            "worker_size=2",
        ),
    ],
)
def test_replay_factory_rejects_unsupported_self_prediction_options(need, match):
    from replay_memory.replay_factory import make_replay_buffer

    with pytest.raises(ValueError, match=match):
        make_replay_buffer(need)


@pytest.mark.parametrize(
    "family_cls",
    [Q_Network_Family, Deteministic_Policy_Gradient_Family],
)
def test_local_families_fail_fast_without_replay_factory(family_cls):
    agent = family_cls.__new__(family_cls)
    agent.replay_factory = None

    with pytest.raises(ValueError, match="ReplayBufferFactory is required"):
        agent.get_memory_setup()


@pytest.mark.parametrize(
    "family_cls",
    [Ape_X_Family, Ape_X_Deteministic_Policy_Gradient_Family],
)
def test_apex_families_fail_fast_without_required_factories(family_cls):
    agent = family_cls.__new__(family_cls)
    agent.multi_replay_factory = None
    agent.worker_replay_factory = FakeWorkerReplayFactory(object())

    with pytest.raises(ValueError, match="MultiPrioritizedReplayBufferFactory is required"):
        agent.get_memory_setup()

    agent.multi_replay_factory = FakeReplayFactory(object())
    agent.worker_replay_factory = None
    with pytest.raises(ValueError, match="WorkerReplayBufferFactory is required"):
        agent.get_memory_setup()


def test_worker_local_replay_factory_seam_builds_without_common_cpprb_import():
    fake_buffer = object()
    factory = FakeWorkerReplayFactory(fake_buffer)
    env_dict = {"obs0": {"shape": [4]}, "done": {}}
    n_s = {"size": 3}

    assert make_worker_local_replay_buffer(factory, 1000, env_dict, n_s) is fake_buffer
    assert factory.calls == [{"local_size": 1000, "env_dict": env_dict, "n_s": n_s}]

    for path in [
        Path("jax_baselines/APE_X/worker.py"),
        Path("jax_baselines/APE_X/dpg_worker.py"),
    ]:
        source = path.read_text()
        assert "jax_baselines.common.cpprb_buffers" not in source
        assert "ReplayBuffer(local_size" not in source
        assert "make_worker_local_replay_buffer" in source
