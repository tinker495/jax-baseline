import numpy as np
import pytest

from env_builder.env_builder import (
    EnvPoolVectorizedEnv,
    GymVectorizedEnv,
    get_env_builder,
)
from replay_memory.cpprb_buffers import NstepReplayBuffer, PrioritizedNstepReplayBuffer
from replay_memory.frame_buffers import FrameStackReplayBuffer


@pytest.mark.parametrize("prioritized", [False, True])
def test_nstep_truncation_projects_zero_termination(prioritized):
    cls = PrioritizedNstepReplayBuffer if prioritized else NstepReplayBuffer
    kwargs = {"alpha": 0.6} if prioritized else {}
    buf = cls(32, {"obs": [1]}, worker_size=1, n_step=3, gamma=0.99, **kwargs)

    for step in range(2):
        obs = np.array([[step]], dtype=np.float32)
        buf.add(
            {"obs": obs},
            np.array([0.0], dtype=np.float32),
            np.float32(1.0),
            {"obs": obs + 1},
            False,
            step == 1,
        )

    assert np.array_equal(buf.sample(2)["terminateds"], np.zeros((2, 1)))


@pytest.mark.parametrize("prioritized", [False, True])
def test_vector_nstep_truncation_stores_zero_termination(prioritized):
    cls = PrioritizedNstepReplayBuffer if prioritized else NstepReplayBuffer
    kwargs = {"alpha": 0.6} if prioritized else {}
    buf = cls(32, {"obs": [1]}, worker_size=2, n_step=3, gamma=0.99, **kwargs)
    action = np.zeros((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    only_first_worker = np.array([True, False])

    for step in range(2):
        obs = np.array([[step], [0]], dtype=np.float32)
        buf.add(
            {"obs": obs},
            action,
            reward,
            {"obs": obs + 1},
            np.zeros(2, dtype=bool),
            np.array([step == 1, False]),
            store_mask=only_first_worker,
        )

    assert np.array_equal(buf.get_buffer()["done"], np.zeros((2, 1)))


def test_vector_nstep_replay_emits_during_long_episode_without_staging_cap():
    buf = NstepReplayBuffer(3000, {"obs": [1]}, worker_size=2, n_step=3, gamma=0.99)
    action = np.zeros((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    no_done = np.zeros(2, dtype=bool)
    only_first_worker = np.array([True, False])

    for step in range(2005):
        obs = np.array([[step], [0]], dtype=np.float32)
        buf.add(
            {"obs": obs},
            action,
            reward,
            {"obs": obs + 1},
            no_done,
            no_done,
            store_mask=only_first_worker,
        )

    assert len(buf) == 2003

    obs = np.array([[2005], [0]], dtype=np.float32)
    buf.add(
        {"obs": obs},
        action,
        reward,
        {"obs": obs + 1},
        np.array([True, False]),
        no_done,
        store_mask=only_first_worker,
    )
    assert len(buf) == 2006


def test_frame_replay_uses_supplied_next_observation_at_truncation():
    buf = FrameStackReplayBuffer(
        16,
        observation_space={"obs": [1, 1, 4]},
        action_space=1,
        n_step=3,
        gamma=0.99,
        n_frames=4,
    )
    obs = np.array([[[[10, 10, 10, 10]]]], dtype=np.uint8)
    mid = np.array([[[[10, 10, 10, 11]]]], dtype=np.uint8)
    boundary_next = np.array([[[[10, 10, 11, 12]]]], dtype=np.uint8)

    buf.add({"obs": obs}, np.float32(0), np.float32(1), {"obs": mid}, False, False)
    buf.add(
        {"obs": mid},
        np.float32(0),
        np.float32(1),
        {"obs": boundary_next},
        False,
        True,
    )

    transition = buf._gather(np.array([0], dtype=np.int64))
    assert np.array_equal(transition["nxtobses"]["obs"][0], boundary_next[0])
    assert transition["terminateds"][0, 0] == 0


def test_gym_vector_fast_path_honors_seed():
    first = GymVectorizedEnv("CartPole-v1", worker_num=2, seed=123)
    second = GymVectorizedEnv("CartPole-v1", worker_num=2, seed=123)
    try:
        assert np.array_equal(first.current_obs()["obs"], second.current_obs()["obs"])
    finally:
        first.close()
        second.close()


def test_single_gym_env_exposes_normalized_continuous_actions():
    builder, _ = get_env_builder("Pendulum-v1")
    env = builder(worker=1, seed=7)
    try:
        assert np.array_equal(env.action_space.low, np.array([-1.0], dtype=np.float32))
        assert np.array_equal(env.action_space.high, np.array([1.0], dtype=np.float32))
        assert np.array_equal(
            env.action(np.array([1.0], dtype=np.float32)),
            np.array([2.0], dtype=np.float32),
        )
    finally:
        env.close()


def test_gym_vector_env_exposes_normalized_continuous_actions():
    env = GymVectorizedEnv("Pendulum-v1", worker_num=2, seed=7)
    try:
        assert np.array_equal(
            env.env.single_action_space.low,
            np.array([-1.0], dtype=np.float32),
        )
        assert np.array_equal(
            env.env.single_action_space.high,
            np.array([1.0], dtype=np.float32),
        )
    finally:
        env.close()


def test_envpool_maps_normalized_actions_to_backend_bounds():
    pytest.importorskip("envpool")
    env = EnvPoolVectorizedEnv("Pendulum-v1", worker_num=2, seed=7)
    try:
        converted = env.action_conv(np.array([[-1.0], [1.0]], dtype=np.float32))
        assert np.array_equal(converted, np.array([[-2.0], [2.0]], dtype=np.float32))
    finally:
        env.close()
