import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.rollout_stats import EpisodeTracker, device_episode_step


def test_device_episode_stats_preserve_lifeloss_dummy_steps_and_batched_logging():
    state = (
        jnp.zeros(2),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.zeros(2, dtype=bool),
    )
    # A life loss, real episode ends, poisoned reset dummies, then same-step resets.
    rewards = jnp.array([[1, 10], [2, 20], [999, 888], [3, 30], [4, 40]], dtype=jnp.float32)
    terminated = jnp.array([[1, 0], [1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool)
    truncated = jnp.array([[0, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=bool)
    autoreset = jnp.array([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]], dtype=bool)
    step_inputs = list(zip(rewards, terminated, truncated, autoreset))
    completed_steps = []
    corrected_rewards = []
    corrected_terminated = []
    with jax.transfer_guard("disallow"):
        for inputs in step_inputs:
            state, reward, terminal, completed = device_episode_step(state, *inputs)
            corrected_rewards.append(reward)
            corrected_terminated.append(terminal)
            completed_steps.append(completed)
        completed_batch = jnp.stack(completed_steps)

    np.testing.assert_array_equal(jnp.stack(corrected_rewards)[2], [0, 0])
    np.testing.assert_array_equal(jnp.stack(corrected_terminated)[2], [True, True])
    np.testing.assert_array_equal(
        jnp.stack(corrected_rewards)[[0, 1, 3, 4], :], rewards[[0, 1, 3, 4], :]
    )
    observed = []
    logs = []
    tracker = EpisodeTracker(lambda *args: logs.append(args), log_interval=2, window=3)
    for step, completed in enumerate(jax.device_get(completed_batch)):
        for done, score, length, timeout in completed:
            if not done:
                continue
            observed.append((step * 2, score, length, timeout))
            tracker.record(
                step * 2,
                episode_reward=score,
                episode_length=length,
                timeout=timeout,
            )
    expected = [
        (0, 1, 1, 0),
        (2, 2, 1, 0),
        (2, 30, 2, 1),
        (6, 3, 1, 0),
        (6, 30, 1, 0),
        (8, 4, 1, 0),
        (8, 40, 1, 0),
    ]
    assert observed == expected
    reference_logs = []
    reference = EpisodeTracker(lambda *args: reference_logs.append(args), log_interval=2, window=3)
    for step, score, length, timeout in expected:
        reference.record(
            step,
            episode_reward=score,
            episode_length=length,
            timeout=timeout,
        )
    assert logs == reference_logs
    assert tracker.describe() == reference.describe()
    for value in state:
        np.testing.assert_array_equal(value, [0, 0])
