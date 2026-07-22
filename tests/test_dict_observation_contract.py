import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces

from env_builder.observations import normalize_observation, normalize_observation_space
from jax_baselines.core.env_protocols import batch_observation
from jax_baselines.core.epoch_buffer import EpochBuffer
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.math.statistics import RunningMeanStd
from replay_memory.cpprb_buffers import ReplayBuffer


def test_observation_contract_normalizes_arrays_and_nested_mappings():
    array = np.arange(3, dtype=np.float32)

    normalized = normalize_observation(array)
    assert list(normalized) == ["obs"]
    assert normalized["obs"] is array

    nested = normalize_observation({"z": array + 2, "a": {"b": array + 1}})
    assert list(nested) == ["a.b", "z"]
    np.testing.assert_array_equal(nested["a.b"], array + 1)

    tuple_observation = normalize_observation((array, {"velocity": array + 3}))
    assert list(tuple_observation) == ["0", "1.velocity"]

    tuple_space = spaces.Tuple(
        (
            spaces.Box(-1, 1, shape=(3,)),
            spaces.Dict({"velocity": spaces.Box(-1, 1, (3,))}),
        )
    )
    assert normalize_observation_space(tuple_space) == {"0": [3], "1.velocity": [3]}


def test_core_batches_observation_dict_without_losing_keys():
    observation = {"position": np.array([1.0, 2.0]), "velocity": np.array([3.0])}

    batched = batch_observation(observation)

    assert list(batched) == ["position", "velocity"]
    assert batched["position"].shape == (1, 2)
    assert batched["velocity"].shape == (1, 1)


def test_core_and_storage_reject_legacy_list_observations():
    with pytest.raises(TypeError, match="observation must be a dict"):
        batch_observation(np.zeros(2, dtype=np.float32))
    with pytest.raises(TypeError, match="shapes must be a dict"):
        RunningMeanStd(shapes=[[2]])
    with pytest.raises(TypeError, match="observation_space must be a dict"):
        ReplayBuffer(4, [[2]])


def test_epoch_and_replay_buffers_keep_observation_keys():
    space = {"position": [2], "velocity": [1]}
    obs = {
        "position": np.array([[1.0, 2.0]], dtype=np.float32),
        "velocity": np.array([[3.0]], dtype=np.float32),
    }
    next_obs = {key: value + 1 for key, value in obs.items()}

    epoch = EpochBuffer(1, space)
    epoch.add(obs, [[0]], [1.0], next_obs, [False], [False])
    epoch_data = epoch.get_buffer()
    assert set(epoch_data["obses"]) == set(space)
    assert epoch_data["obses"]["position"].shape == (1, 1, 2)

    replay = ReplayBuffer(4, space)
    replay.add(obs, [0], 1.0, next_obs, False)
    replay_data = replay.sample(1)
    assert set(replay_data["obses"]) == set(space)
    assert set(replay_data["nxtobses"]) == set(space)


def test_crossq_critic_loss_concatenates_dict_observation_values():
    agent = CrossQ.__new__(CrossQ)
    agent._gamma = 0.99
    seen = {}

    def preproc(_params, _key, observations):
        seen.update(observations)
        return observations["obs"]

    agent.preproc = preproc
    agent._get_pi_log_prob = lambda _params, features, _key: (
        jnp.zeros((features.shape[0], 1)),
        jnp.zeros((features.shape[0], 1)),
    )
    agent.critic = lambda params, _key, features, _actions, _training: (
        (jnp.zeros((features.shape[0], 1)), jnp.zeros((features.shape[0], 1))),
        {"batch_stats": params["batch_stats"]},
    )
    observations = {"obs": jnp.ones((2, 3))}
    next_observations = {"obs": jnp.full((2, 3), 2.0)}

    loss, _ = agent._critic_loss(
        {"batch_stats": {}},
        {},
        observations,
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        next_observations,
        jnp.ones((2, 1)),
        0.1,
        jnp.ones((2, 1)),
        None,
    )

    assert jnp.isfinite(loss)
    np.testing.assert_array_equal(
        seen["obs"],
        jnp.concatenate([observations["obs"], next_observations["obs"]]),
    )
