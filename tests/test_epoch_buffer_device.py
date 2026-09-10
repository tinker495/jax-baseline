import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.core.epoch_buffer import EpochBuffer
from jax_baselines.core.eval import _normalize_action_for_step


def test_epoch_buffer_keeps_worker_time_order_and_snapshots_across_rollouts():
    buffer = EpochBuffer(3, {"actor_obs": [2], "critic_obs": [1]}, 2, [1])
    assert buffer.memory_backend == "cpu"
    for _ in range(2):
        observations = []
        for step in range(3):
            obs = {
                "actor_obs": np.full((2, 2), step, np.float32),
                "critic_obs": np.array([[step], [10 + step]], np.float32),
            }
            observations.append({key: value.copy() for key, value in obs.items()})
            action = np.array([[step], [step + 1]], np.float32)
            reward = np.array([step, step + 2], np.float32)
            terminated = np.array([False, True])
            truncated = np.array([True, False])
            buffer.add(obs, action, reward, obs, terminated, truncated)
            for value in obs.values():
                value.fill(-99)
            action.fill(-99)
            reward.fill(-99)
            terminated.fill(False)
            truncated.fill(False)
        with pytest.raises(ValueError, match="full"):
            buffer.add({}, [], [], {}, [], [])
        batch = buffer.get_buffer()
        assert all(isinstance(value, np.ndarray) for value in jax.tree.leaves(batch))
        for key in observations[0]:
            np.testing.assert_array_equal(
                batch["obses"][key], np.stack([o[key] for o in observations], axis=1)
            )
        np.testing.assert_array_equal(batch["actions"], [[[0], [1], [2]], [[1], [2], [3]]])
        np.testing.assert_array_equal(batch["rewards"], [[0, 1, 2], [2, 3, 4]])
        np.testing.assert_array_equal(batch["terminateds"], [[False] * 3, [True] * 3])
        np.testing.assert_array_equal(batch["truncateds"], [[True] * 3, [False] * 3])
        with pytest.raises(ValueError, match="empty"):
            buffer.get_buffer()


@pytest.fixture
def gpu_device():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("JAX GPU backend is unavailable")
    if not devices:
        pytest.skip("JAX GPU backend is unavailable")
    return devices[0]


def test_epoch_buffer_accepts_device_arrays_without_host_transfers(gpu_device):
    buffer = EpochBuffer(2, {"unified_obs": [2]}, 2, [1], memory_backend="gpu")
    with jax.default_device(gpu_device):
        obs = {"unified_obs": jnp.arange(4, dtype=jnp.float32).reshape(2, 2)}
        action, reward, done = jnp.ones((2, 1)), jnp.ones(2), jnp.zeros(2, dtype=bool)
    with jax.transfer_guard("disallow"):
        for _ in range(2):
            buffer.add(obs, action, reward, obs, done, done)
        batch = buffer.get_buffer()
        jax.block_until_ready(batch)
    assert batch["obses"]["unified_obs"].shape == (2, 2, 2)
    assert len(jax.tree.leaves(batch)) == 6
    assert all(value.devices() == {gpu_device} for value in jax.tree.leaves(batch))


def test_continuous_evaluation_action_preserves_device_array():
    action = jnp.ones((1, 12), dtype=jnp.float32)
    with jax.transfer_guard("disallow"):
        normalized = _normalize_action_for_step(action)
        jax.block_until_ready(normalized)
    assert isinstance(normalized, jax.Array)
    assert normalized.shape == (12,)


def test_gpu_epoch_memory_ignores_cpu_default_device_and_snapshots_host_inputs(
    gpu_device,
):
    with jax.default_device(jax.devices("cpu")[0]):
        buffer = EpochBuffer(1, {"unified_obs": [2]}, 2, [1], memory_backend="gpu")
        obs = {"unified_obs": np.arange(4, dtype=np.float32).reshape(2, 2)}
        buffer.add(obs, [[1], [2]], [3, 4], obs, [False, True], [True, False])
        obs["unified_obs"].fill(-99)
        batch = buffer.get_buffer()
        jax.block_until_ready(batch)
    assert buffer.memory_backend == "gpu"
    assert all(value.devices() == {gpu_device} for value in jax.tree.leaves(batch))
    np.testing.assert_array_equal(batch["obses"]["unified_obs"], [[[0, 1]], [[2, 3]]])
    np.testing.assert_array_equal(batch["rewards"], [[3], [4]])
    assert batch["terminateds"].dtype == np.bool_


def test_gpu_epoch_memory_fails_clearly_when_gpu_is_unavailable(monkeypatch):
    def unavailable_devices(backend):
        assert backend == "gpu"
        raise RuntimeError("GPU backend unavailable")

    monkeypatch.setattr(jax, "devices", unavailable_devices)
    with pytest.raises(RuntimeError, match="GPU epoch memory requires an available JAX GPU"):
        EpochBuffer(1, {"unified_obs": [2]}, memory_backend="gpu")
    assert EpochBuffer(1, {"unified_obs": [2]}).memory_backend == "cpu"


@pytest.mark.parametrize("memory_backend", ["auto", "tpu", None])
def test_epoch_memory_rejects_unresolved_or_invalid_backend(memory_backend):
    with pytest.raises(ValueError, match="memory_backend must be"):
        EpochBuffer(1, {"unified_obs": [2]}, memory_backend=memory_backend)
