import numpy as np

from replay_memory.transition_buffers import TransitionReplayBuffer


def test_self_prediction_filled_mask_keeps_terminal_reward_step(monkeypatch):
    buf = TransitionReplayBuffer(10, observation_space=[[1]], action_space=1, prediction_depth=3)
    action = np.array([0], dtype=np.float32)

    buf.add(
        [np.array([0], dtype=np.float32)],
        action,
        1.0,
        [np.array([1], dtype=np.float32)],
        False,
    )
    buf.add(
        [np.array([1], dtype=np.float32)],
        action,
        2.0,
        [np.array([2], dtype=np.float32)],
        True,
    )

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: np.array([0, 1]))

    data = buf.sample(2)
    terminated = data["terminateds"]
    filled = data["filled"]

    assert terminated.tolist() == [[False, True, True], [True, True, True]]
    assert filled.tolist() == [[True, True, False], [True, False, False]]
    np.testing.assert_allclose((data["rewards"] * filled).sum(axis=1), np.array([3.0, 2.0]))
