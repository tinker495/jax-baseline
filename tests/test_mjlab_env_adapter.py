import builtins
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env_builder.env_builder import get_env_builder
from env_builder.mjlab_env import MjlabSingleEnv, MjlabVectorizedEnv, make_mjlab_env
from jax_baselines.core.env_protocols import SingleEnv, VectorizedEnv

torch = pytest.importorskip("torch")


class FakeEnv:
    def __init__(self, cfg, device, render_mode=None):
        self.render_mode = render_mode
        self.metadata = {"render_fps": 50}
        self.cfg, self.device = cfg, device
        self.num_envs = cfg.scene.num_envs
        self.single_action_space = SimpleNamespace(shape=(1,))
        self.obs = torch.zeros((self.num_envs, 1))
        self.critic_obs = torch.zeros((self.num_envs, 2))
        self.rewards = torch.zeros(self.num_envs)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.truncated = self.terminated.clone()
        self.resets, self.closed = [], 0

    def reset(self, *, seed=None, env_ids=None):
        self.resets.append((seed, env_ids))
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self.obs[ids] = 0 if seed is None else seed
        self.critic_obs[ids] = 100 if seed is None else seed + 100
        self.rewards[ids] = 0
        self.terminated[ids] = False
        self.truncated[ids] = False
        return {
            "actor": {"state": self.obs},
            "critic": self.critic_obs,
            "command": self.obs + 5,
        }, {}

    def step(self, actions):
        self.obs += 1
        self.critic_obs += 10
        self.rewards[:] = actions[:, 0]
        self.terminated[:] = actions[:, 0] < 0
        self.truncated[:] = actions[:, 0] > 1
        return (
            {"actor": {"state": self.obs}, "critic": self.critic_obs, "command": self.obs + 5},
            self.rewards,
            self.terminated,
            self.truncated,
            {"metric": self.rewards},
        )

    def render(self):
        if self.render_mode is None:
            return None
        return np.full((16, 16, 3), int(self.obs[0, 0]), dtype=np.uint8)

    def close(self):
        self.closed += 1


def install_runtime(monkeypatch):
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        seed=None,
        auto_reset=True,
        sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.002)),
        decimation=10,
        episode_length_s=20,
    )
    modules = {
        "mjlab": {},
        "mjlab.tasks": {},
        "mjlab.envs": {"ManagerBasedRlEnv": FakeEnv},
        "mjlab.tasks.registry": {"load_env_cfg": lambda task: cfg},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return cfg


@pytest.mark.parametrize("jax_arrays", [False, True])
def test_vector_factory_and_terminal_snapshots_survive_partial_reset(monkeypatch, jax_arrays):
    cfg = install_runtime(monkeypatch)
    env = make_mjlab_env("task", 3, seed=7, episode_length=5, device="cpu", jax_arrays=jax_arrays)
    assert isinstance(env, MjlabVectorizedEnv)
    assert VectorizedEnv in type(env).__mro__
    assert cfg.scene.num_envs == 3 and cfg.seed == 7 and not cfg.auto_reset
    assert cfg.episode_length_s == pytest.approx(0.1)
    assert env.env_info["observation_space"] == {
        "actor_state": [1],
        "critic_obs": [2],
        "unified_command": [1],
    }
    assert env.env_info["action_size"] == [1]
    with pytest.raises(RuntimeError, match="without a preceding"):
        env.get_result()
    with pytest.raises(ValueError, match="shape"):
        env.step(np.ones((3, 2)))
    env.step(np.array([[-1], [2], [1]]))
    with pytest.raises(RuntimeError, match="previous result"):
        env.step(np.ones((3, 1)))
    with pytest.raises(RuntimeError, match="in flight"):
        env.reset()
    obs, reward, terminated, truncated, info = env.get_result()
    assert isinstance(obs["actor_state"], jax.Array if jax_arrays else np.ndarray)
    np.testing.assert_array_equal(obs["actor_state"], [[8], [8], [8]])
    np.testing.assert_array_equal(env.current_obs()["actor_state"], [[0], [0], [8]])
    np.testing.assert_array_equal(obs["critic_obs"], [[117, 117]] * 3)
    np.testing.assert_array_equal(
        env.current_obs()["critic_obs"], [[100, 100], [100, 100], [117, 117]]
    )
    np.testing.assert_array_equal(obs["unified_command"], [[13], [13], [13]])
    np.testing.assert_array_equal(env.current_obs()["unified_command"], [[5], [5], [13]])
    np.testing.assert_array_equal(reward, [-1, 2, 1])
    np.testing.assert_array_equal(info["metric"], reward)
    np.testing.assert_array_equal(terminated, [True, False, False])
    np.testing.assert_array_equal(truncated, [False, True, False])
    np.testing.assert_array_equal(env.env.resets[-1][1], [0, 1])
    np.testing.assert_array_equal(
        env.real_reset_mask(terminated, truncated, {}), [True, True, False]
    )
    assert not env.autoreset_mask(terminated, truncated, {}).any()
    env.step(jnp.zeros((3, 1)) if jax_arrays else np.zeros((3, 1)))
    env.get_result()
    np.testing.assert_array_equal(obs["actor_state"], [[8], [8], [8]])
    np.testing.assert_array_equal(reward, [-1, 2, 1])
    np.testing.assert_array_equal(info["metric"], [-1, 2, 1])
    env.close()
    env.close()
    assert env.env.closed == 1


def test_jax_exchange_does_not_convert_torch_tensors_to_cpu_or_numpy(monkeypatch):
    install_runtime(monkeypatch)

    def host_conversion(*args, **kwargs):
        raise AssertionError("JAX tensor exchange must stay on the device")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", host_conversion)
        guard.setattr(torch.Tensor, "numpy", host_conversion)
        env = make_mjlab_env("task", 2, device="cpu", jax_arrays=True)
        assert isinstance(env, MjlabVectorizedEnv)
        env.step(jnp.array([[-1.0], [1.0]]))
        obs, reward, terminated, truncated, info = env.get_result()
        assert all(
            isinstance(value, jax.Array)
            for value in (*obs.values(), reward, terminated, truncated, info["metric"])
        )
        env.close()


def test_single_reset_cache_explicit_seed_and_default_duration(monkeypatch):
    cfg = install_runtime(monkeypatch)
    env = make_mjlab_env("task", seed=3, device="cpu")
    assert isinstance(env, MjlabSingleEnv)
    assert SingleEnv in type(env).__mro__
    assert cfg.episode_length_s == 20
    np.testing.assert_array_equal(env.reset()[0]["actor_state"], [3])
    assert len(env._vector.env.resets) == 1
    obs, reward, terminated, truncated, _ = env.step(np.array([-1]))
    assert reward == -1 and terminated and not truncated
    np.testing.assert_array_equal(obs["actor_state"], [4])
    np.testing.assert_array_equal(env.reset()[0]["actor_state"], [0])
    assert len(env._vector.env.resets) == 2
    np.testing.assert_array_equal(env.reset(seed=9)[0]["actor_state"], [9])
    np.testing.assert_array_equal(env.reset()[0]["actor_state"], [0])
    env.close()


@pytest.mark.parametrize(
    ("selector", "key", "shape"),
    [("actor", "unified_actor.state", [1]), ("critic", "unified_critic", [2])],
)
def test_explicit_group_selection_is_shared_by_actor_and_critic(monkeypatch, selector, key, shape):
    install_runtime(monkeypatch)
    env = make_mjlab_env("task", 2, observation_key=selector, device="cpu")
    assert isinstance(env, MjlabVectorizedEnv)
    assert env.observation_space == {key: shape}
    assert list(env.current_obs()) == [key]
    env.close()


def test_mjlab_without_separate_groups_uses_unified_observations(monkeypatch):
    install_runtime(monkeypatch)
    env = make_mjlab_env("task", 2, device="cpu")
    assert isinstance(env, MjlabVectorizedEnv)
    selected = env._selected({"actor": {"state": env.env.obs}})
    assert list(selected) == ["unified_actor.state"]
    env.close()


def test_single_factory_metadata_preserves_canonical_keys(monkeypatch):
    install_runtime(monkeypatch)
    for env_name, backend, expected in [
        ("CartPole-v1", "gymnasium", {"unified_obs": [4]}),
        ("task", "mjlab", {"actor_state": [1], "critic_obs": [2], "unified_command": [1]}),
    ]:
        builder, _ = get_env_builder(env_name, env_backend=backend, device="cpu")
        prepared = builder.prepare_envs(num_workers=1, seed=3)
        try:
            assert prepared.env_info["observation_space"] == expected
            assert set(prepared.env.reset()[0]) == set(expected)
            assert set(prepared.eval_env.reset()[0]) == set(expected)
        finally:
            prepared.env.close()
            prepared.eval_env.close()


def test_factory_validation_and_initialization_cleanup(monkeypatch):
    with pytest.raises(ValueError, match="render_mode"):
        make_mjlab_env("task", render_mode="human")
    with pytest.raises(ValueError, match="worker_num"):
        make_mjlab_env("task", 0)
    with pytest.raises(ValueError, match="episode_length"):
        make_mjlab_env("task", episode_length=0)
    install_runtime(monkeypatch)
    closed = []
    monkeypatch.setattr(FakeEnv, "close", lambda self: closed.append(True))
    with pytest.raises(KeyError, match="missing"):
        make_mjlab_env("task", observation_key="missing", device="cpu")
    assert closed == [True]


def test_missing_optional_runtime_has_install_hint(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "torch" or name.startswith("mjlab"):
            raise ImportError("missing runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="'mjlab' extra"):
        make_mjlab_env("task")


def test_recording_preserves_terminal_frame_until_the_next_reset(monkeypatch):
    install_runtime(monkeypatch)
    env = make_mjlab_env("task", seed=3, device="cpu", render_mode="rgb_array")
    assert isinstance(env, MjlabSingleEnv)
    assert env.render_mode == "rgb_array" and env.metadata["render_fps"] == 50
    env.reset()
    assert (env.render() == 3).all()
    env.step(np.array([-1]))
    assert (env.render() == 4).all()  # The simulator has already reset to zero.
    env.reset()
    assert (env.render() == 0).all()
    env.close()


def test_training_does_not_render(monkeypatch):
    install_runtime(monkeypatch)

    def unexpected_render(self):
        raise AssertionError("Training must not render")

    monkeypatch.setattr(FakeEnv, "render", unexpected_render)
    env = make_mjlab_env("task", worker_num=2, device="cpu")
    assert isinstance(env, MjlabVectorizedEnv)
    env.step(np.zeros((2, 1)))
    env.get_result()
    env.close()
