import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.rollout_stats import EpisodeTracker, device_episode_step


def test_device_episode_stats_preserve_lifeloss_dummy_steps_and_batched_logging():
    state = (
        jnp.zeros(2),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.zeros(2),
        jnp.zeros(2, dtype=bool),
        jnp.zeros(2, dtype=bool),
    )
    # A life loss, real episode ends, poisoned reset dummies, then same-step resets.
    rewards = jnp.array([[1, 10], [2, 20], [999, 888], [3, 30], [4, 40]], dtype=jnp.float32)
    terminated = jnp.array([[1, 0], [1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool)
    truncated = jnp.array([[0, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=bool)
    real_reset = jnp.array([[0, 0], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=bool)
    autoreset = jnp.array([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]], dtype=bool)
    original = jnp.array(
        [[100, 1000], [200, 2000], [99999, 88888], [300, 0], [0, 0]], dtype=jnp.float32
    )
    original_present = jnp.array([[1, 1], [1, 1], [1, 1], [1, 0], [0, 0]], dtype=bool)
    step_inputs = list(
        zip(rewards, terminated, truncated, real_reset, autoreset, original, original_present)
    )
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
        for done, score, length, timeout, original_score, emit_original in completed:
            if not done:
                continue
            observed.append(
                (step * 2, score, length, timeout, original_score if emit_original else None)
            )
            tracker.record(
                step * 2,
                episode_reward=score,
                episode_length=length,
                timeout=timeout,
                original_reward=original_score if emit_original else None,
            )
    expected = [
        (0, 1, 1, 0, None),
        (2, 2, 1, 0, 300),
        (2, 30, 2, 1, 3000),
        (6, 3, 1, 0, 300),
        (6, 30, 1, 0, None),
        (8, 4, 1, 0, None),
        (8, 40, 1, 0, None),
    ]
    assert observed == expected
    reference_logs = []
    reference = EpisodeTracker(lambda *args: reference_logs.append(args), log_interval=2, window=3)
    for step, score, length, timeout, original_score in expected:
        reference.record(
            step,
            episode_reward=score,
            episode_length=length,
            timeout=timeout,
            original_reward=original_score,
        )
    assert logs == reference_logs
    assert tracker.describe() == reference.describe()
    for value in state:
        np.testing.assert_array_equal(value, [0, 0])
