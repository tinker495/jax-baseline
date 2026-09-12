import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.core.env_protocols import PreparedEnvSpec
from jax_baselines.core.normalization import normalize_empirical_observation
from jax_baselines.core.runtime_adapters import NoOpLoggerRun
from jax_baselines.core.training_session import RunContext


class _Env:
    action_space = object()

    def __init__(self, observation):
        self.observation_space = {"unified_obs": [2]}
        self.observation = observation
        self.resets = 0

    def prepare_envs(self, num_workers=1, seed=None):
        return PreparedEnvSpec(
            self,
            _Env(self.observation),
            {
                "observation_space": self.observation_space,
                "action_size": [1],
                "action_type": "continuous",
                "env_type": "single",
                "env_id": "MemoryBackend-v0",
                "worker_num": 1,
                "core_env_type": "SingleEnv",
                "runtime": {
                    "backend": "fake",
                    "backend_env_id": "MemoryBackend-v0",
                    "seed": seed,
                    "seed_rule": "constructor seed",
                    "reward_clipping": "none",
                    "episodic_life": False,
                },
            },
        )

    def reset(self):
        self.resets += 1
        return {"unified_obs": self.observation}, {}

    def step(self, action):
        return {"unified_obs": self.observation}, 1.0, False, False, {}

    def close(self):
        pass


@pytest.mark.parametrize("backend", ["auto", "cpu"])
@pytest.mark.parametrize("jax_observation", [False, True])
def test_host_environment_keeps_rollout_arrays_on_cpu(backend, jax_observation):
    observation = np.array([1.0, 2.0], np.float32)
    if jax_observation:
        observation = jax.device_put(observation, jax.devices("cpu")[0])
    agent = Actor_Critic_Policy_Gradient_Family(
        _Env(observation),
        None,
        memory_backend=backend,
        obs_rms_norm=True,
        optimizer_factory=optax.sgd,
        _init_setup_model=False,
    )
    agent.get_memory_setup()
    assert agent.memory_backend == "cpu"
    assert agent.buffer.memory_backend == "cpu"
    assert agent.obs_rms is not None
    assert all(isinstance(value, np.ndarray) for value in agent.obs_rms.means.values())
    normalized = normalize_empirical_observation(
        {"unified_obs": observation[None, :]},
        agent.obs_rms,
        on_device=agent.memory_backend == "gpu",
    )
    assert isinstance(normalized["unified_obs"], np.ndarray)
    agent._get_actions = lambda params, obs: (jnp.zeros((1, 1)), jnp.ones((1, 1)))
    assert isinstance(agent.action_continuous(normalized), np.ndarray)
    agent._get_actions = lambda params, obs: jnp.array([[0.25, 0.75]])
    assert isinstance(agent.action_discrete(normalized), np.ndarray)


@pytest.mark.parametrize("backend", ["auto", "gpu", "cpu"])
def test_gpu_environment_selects_device_memory_unless_cpu_requested(backend):
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("Requires a JAX GPU")
    agent = Actor_Critic_Policy_Gradient_Family(
        _Env(jax.device_put(np.ones(2, np.float32), gpu_devices[0])),
        None,
        memory_backend=backend,
        obs_rms_norm=True,
        optimizer_factory=optax.sgd,
        _init_setup_model=False,
    )
    agent.get_memory_setup()
    expected = "cpu" if backend == "cpu" else "gpu"
    assert agent.memory_backend == expected
    assert agent.buffer.memory_backend == expected
    assert agent.obs_rms is not None
    if expected == "gpu":
        mean = agent.obs_rms.means["unified_obs"]
        assert isinstance(mean, jax.Array)
        assert all(device.platform == "gpu" for device in mean.devices())
    else:
        assert isinstance(agent.obs_rms.means["unified_obs"], np.ndarray)


def test_forced_gpu_fails_without_gpu():
    if any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Requires a CPU-only JAX runtime")
    with pytest.raises(ValueError, match="GPU|gpu"):
        agent = Actor_Critic_Policy_Gradient_Family(
            _Env(np.ones(2, np.float32)),
            None,
            memory_backend="gpu",
            optimizer_factory=optax.sgd,
            _init_setup_model=False,
        )
        agent.get_memory_setup()


@pytest.mark.parametrize("backend", ["numpy", "cuda"])
def test_invalid_memory_backend_fails_at_construction(backend):
    with pytest.raises(ValueError, match="memory_backend"):
        Actor_Critic_Policy_Gradient_Family(
            _Env(np.ones(2, np.float32)),
            None,
            memory_backend=backend,
            optimizer_factory=optax.sgd,
            _init_setup_model=False,
        )


def test_auto_detection_reuses_single_environment_reset_in_first_rollout():
    env = _Env(np.array([7.0, 11.0], np.float32))
    agent = Actor_Critic_Policy_Gradient_Family(
        env,
        None,
        batch_size=3,
        optimizer_factory=optax.sgd,
        _init_setup_model=False,
    )
    agent.get_memory_setup()
    agent.actions = lambda obs: np.zeros((1, 1), np.float32)

    agent.learn_SingleEnv(
        RunContext(
            pbar=range(1, 2),
            logger_run=NoOpLoggerRun("/tmp"),
            log_interval=100,
            eval_freq=100,
        )
    )

    assert env.resets == 1
    np.testing.assert_array_equal(
        agent.buffer.get_buffer()["obses"]["unified_obs"], [[[7.0, 11.0]]]
    )
