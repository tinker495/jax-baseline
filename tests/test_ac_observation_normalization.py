from types import SimpleNamespace

import numpy as np
import pytest

from experiments.checkpoint_store import FileCheckpointStore
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.core.normalization import (
    RunningMeanStd,
    normalize_empirical_observation,
)
from jax_baselines.core.runtime_adapters import NoOpLoggerRun
from jax_baselines.core.training_session import RunContext


def _agent(enabled=True):
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.memory_backend = "cpu"
    agent.memory_device = None
    agent.action_type = "continuous"
    agent._initial_reset = None
    agent.observation_space = {"unified_obs": [1]}
    agent.obs_rms_norm = enabled
    agent.obs_rms = (
        RunningMeanStd(epsilon=0.0, shapes=agent.observation_space, dtype=np.float32)
        if enabled
        else None
    )
    agent.actor_params = np.float32(2.0)
    agent.critic_params = np.float32(3.0)
    agent._get_actions = lambda params, obs: (
        params * obs["unified_obs"],
        np.ones_like(obs["unified_obs"]),
        np.zeros_like(obs["unified_obs"]),
    )
    agent.actions = agent.action_continuous
    return agent


def test_normalization_preserves_feature_roles_and_freezes_statistics():
    agent = _agent()
    agent.obs_rms = RunningMeanStd(
        epsilon=0.0, shapes={"actor_obs": [2], "critic_obs": [1]}, dtype=np.float32
    )
    agent.obs_rms.update(
        {
            "actor_obs": np.array([[1, 10], [3, 14]], dtype=np.float32),
            "critic_obs": np.array([[100], [300]], dtype=np.float32),
        }
    )
    obs = {
        "actor_obs": np.array([[3, 14]], dtype=np.float32),
        "critic_obs": np.array([[300]], dtype=np.float32),
    }

    normalized = normalize_empirical_observation(
        obs, agent.obs_rms, on_device=agent.memory_backend == "gpu"
    )

    np.testing.assert_allclose(normalized["actor_obs"], [[1 / 1.01, 2 / 2.01]])
    np.testing.assert_allclose(normalized["critic_obs"], [[100 / 100.01]])
    np.testing.assert_array_equal(obs["actor_obs"], [[3, 14]])
    assert agent.obs_rms.count == 2
    assert all(value.dtype == np.float32 for value in normalized.values())


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_and_recording_restore_checkpoint_normalization(tmp_path, monkeypatch, enabled):
    agent = _agent(enabled)
    if agent.obs_rms is not None:
        agent.obs_rms.update({"unified_obs": np.array([[2], [4]], dtype=np.float32)})
    agent.checkpoint_store = FileCheckpointStore()
    agent.save_params(str(tmp_path))
    restored = _agent(not enabled)
    restored.checkpoint_store = FileCheckpointStore()
    restored.load_params(str(tmp_path))
    restored.eval_env = {"unified_obs": np.array([[5]], dtype=np.float32)}
    restored.eval_eps = 1
    restored.conv_action = lambda action: action
    restored.env_builder = restored.eval_env
    restored.record_test_fn = lambda builder, logger, actions, episode, conv_action: actions(
        builder
    )
    monkeypatch.setattr(
        "jax_baselines.A2C.base_class.evaluate_policy",
        lambda env, episodes, actions, **kwargs: actions(env),
    )

    evaluated = restored.eval(SimpleNamespace(logger_run=None), 0)
    recorded = restored.test_eval_env(None, 1)

    expected = [[4 / 1.01]] if enabled else [[10]]
    np.testing.assert_allclose(evaluated, expected)
    np.testing.assert_array_equal(recorded, evaluated)
    assert restored.obs_rms_norm is enabled
    if enabled:
        assert restored.obs_rms is not None
        assert restored.obs_rms.count == 2
        np.testing.assert_array_equal(restored.obs_rms.means["unified_obs"], [3])
        np.testing.assert_array_equal(restored.obs_rms.vars["unified_obs"], [1])
    else:
        assert restored.obs_rms is None


class _SingleEnv:
    def __init__(self, reset_value):
        self.reset_value = reset_value
        self.resets = 0
        self.steps = 0

    def reset(self):
        self.resets += 1
        return {
            "unified_obs": np.array([10 if self.resets == 1 else self.reset_value], np.float32)
        }, {}

    def step(self, action):
        self.steps += 1
        return (
            {"unified_obs": np.array([100 if self.steps == 1 else 4], np.float32)},
            1.0,
            False,
            self.steps == 1,
            {},
        )


class _VectorEnv:
    def __init__(self, reset_value):
        self.reset_value = reset_value
        self.steps = 0

    def current_obs(self):
        return {
            "unified_obs": np.array(
                [[[10], [12]], [[self.reset_value], [4]], [[6], [8]]][self.steps],
                np.float32,
            )
        }

    def step(self, actions):
        self.steps += 1

    def get_result(self):
        return (
            {"unified_obs": np.array([[100], [4]] if self.steps == 1 else [[6], [8]], np.float32)},
            np.ones(2, np.float32),
            np.zeros(2, bool),
            np.array([self.steps == 1, False]),
            {},
        )

    def autoreset_mask(self, terminateds, truncateds, infos):
        return np.zeros(2, bool)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("reset_value", [2.0, 1000.0])
@pytest.mark.parametrize("initial_values", [(), (2.0, 4.0)])
def test_rollout_normalizes_successors_before_reset_statistics(
    vectorized, reset_value, initial_values
):
    agent = _agent()
    assert agent.obs_rms is not None
    if initial_values:
        agent.obs_rms.update({"unified_obs": np.array(initial_values, np.float32)[:, None]})
    initial_mean = agent.obs_rms.means["unified_obs"].copy()
    initial_var = agent.obs_rms.vars["unified_obs"].copy()
    agent.env = _VectorEnv(reset_value) if vectorized else _SingleEnv(reset_value)
    agent.env_type = "VectorizedEnv" if vectorized else "SingleEnv"
    agent.worker_size = 2 if vectorized else 1
    agent.batch_size = 2
    agent.action_type = "continuous"
    agent.actions = lambda obs: obs["unified_obs"]
    agent.conv_action = lambda action: action
    transitions = []
    train_counts = []
    agent.buffer = SimpleNamespace(
        add=lambda *transition, old_policy=None: transitions.append(transition)
    )
    agent.train_step = lambda steps, logger_run: train_counts.append(agent.obs_rms.count) or 0.0
    agent.eval = lambda ctx, steps: None
    ctx = RunContext(
        pbar=range(0, 2 * agent.worker_size, agent.worker_size),
        logger_run=NoOpLoggerRun("/tmp"),
        log_interval=10_000,
        eval_freq=10_000,
    )

    agent.run_training_loop(ctx)

    assert agent.obs_rms is not None
    assert agent.obs_rms.count == len(initial_values) + 2 * agent.worker_size
    assert train_counts == [len(initial_values) + 2 * agent.worker_size]
    np.testing.assert_allclose(
        transitions[0][0]["unified_obs"],
        (np.array([[10], [12]] if vectorized else [[10]]) - initial_mean)
        / (np.sqrt(initial_var) + 0.01),
    )
    np.testing.assert_allclose(
        transitions[0][3]["unified_obs"],
        (np.array([[100], [4]] if vectorized else [[100]]) - initial_mean)
        / (np.sqrt(initial_var) + 0.01),
    )
    first_action_values = np.array(
        [*initial_values, reset_value, 4] if vectorized else [*initial_values, reset_value],
        np.float32,
    )
    np.testing.assert_allclose(
        transitions[1][0]["unified_obs"],
        (
            np.array([[reset_value], [4]] if vectorized else [[reset_value]])
            - np.mean(first_action_values)
        )
        / (np.std(first_action_values) + 0.01),
    )
    np.testing.assert_allclose(
        transitions[1][3]["unified_obs"],
        (np.array([[6], [8]] if vectorized else [[4]]) - np.mean(first_action_values))
        / (np.std(first_action_values) + 0.01),
    )
    action_values = np.concatenate((first_action_values, [6, 8] if vectorized else [4]))
    np.testing.assert_allclose(agent.obs_rms.means["unified_obs"], [np.mean(action_values)])
    np.testing.assert_allclose(agent.obs_rms.vars["unified_obs"], [np.var(action_values)])
    assert transitions[0][5][0]
