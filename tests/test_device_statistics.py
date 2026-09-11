import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.core.normalization import RunningMeanStd


def test_host_statistics_keep_jax_observations_on_numpy_boundary():
    host = RunningMeanStd(epsilon=0.0, shapes={"actor_obs": (2,)}, on_device=False)
    observations = {"actor_obs": jnp.array([[1.0, 10.0], [3.0, 14.0]])}

    host.update(observations)
    normalized = host.normalize(observations)

    assert isinstance(host.means["actor_obs"], np.ndarray)
    assert isinstance(host.vars["actor_obs"], np.ndarray)
    assert isinstance(normalized["actor_obs"], np.ndarray)
    assert not isinstance(host.count, jax.Array)
    np.testing.assert_array_equal(host.means["actor_obs"], [2.0, 12.0])
    np.testing.assert_array_equal(host.vars["actor_obs"], [1.0, 4.0])
    np.testing.assert_allclose(normalized["actor_obs"], [[-1.0, -1.0], [1.0, 1.0]], atol=1e-8)


@pytest.mark.parametrize("epsilon", [0.0, 1e-4])
def test_device_statistics_match_numpy_and_restore_without_runtime_transfers(epsilon):
    shapes = {"actor_obs": (3,), "critic_obs": (2,)}
    host = RunningMeanStd(epsilon=epsilon, shapes=shapes, dtype=np.float32)
    device = RunningMeanStd(epsilon=epsilon, shapes=shapes, on_device=True)
    rng = np.random.default_rng(7)
    for batch_size in (7, 3, 11):
        observations = {
            key: rng.normal(size=(batch_size, *shape)).astype(np.float32)
            for key, shape in shapes.items()
        }
        device_observations = jax.tree.map(jnp.asarray, observations)
        host.update(observations)
        with jax.transfer_guard("disallow"):
            device.update(device_observations)
            normalized = device.normalize(device_observations)
        for key in shapes:
            assert isinstance(device.means[key], jax.Array)
            assert isinstance(device.vars[key], jax.Array)
            assert isinstance(normalized[key], jax.Array)
            np.testing.assert_allclose(device.means[key], host.means[key], atol=1e-6)
            np.testing.assert_allclose(device.vars[key], host.vars[key], rtol=1e-5)
            np.testing.assert_allclose(
                normalized[key], host.normalize(observations)[key], atol=1e-6
            )
    assert isinstance(device.count, jax.Array)
    assert device.count.dtype == jnp.float32
    assert float(device.count) == pytest.approx(host.count)
    restored = RunningMeanStd.from_state(device.to_state(), on_device=True)
    with jax.transfer_guard("disallow"):
        restored.update(device_observations)
        device.update(device_observations)
    for key in shapes:
        np.testing.assert_array_equal(restored.means[key], device.means[key])
        np.testing.assert_array_equal(restored.vars[key], device.vars[key])
    np.testing.assert_array_equal(restored.count, device.count)


@pytest.mark.parametrize(
    "observations",
    [
        {"actor_obs": np.zeros((2, 3)), "critic_obs": np.zeros((3, 2))},
        {"actor_obs": np.zeros((2, 4)), "critic_obs": np.zeros((2, 2))},
        {"actor_obs": np.zeros((0, 3)), "critic_obs": np.zeros((0, 2))},
        {"actor_obs": np.zeros((2, 3))},
    ],
)
def test_device_statistics_validate_batch_before_updating(observations):
    device = RunningMeanStd(
        epsilon=0.0, shapes={"actor_obs": (3,), "critic_obs": (2,)}, on_device=True
    )
    with pytest.raises(ValueError, match="Observation batch"):
        device.update(observations)
    assert float(device.count) == 0.0
