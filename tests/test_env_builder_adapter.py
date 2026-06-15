"""Regression coverage for the env_builder adapter migration."""

from __future__ import annotations

import importlib

import numpy as np

from jax_baselines.core.env_info import get_local_env_info, infer_action_meta
from jax_baselines.core.env_protocols import Env, EnvInfo, SingleEnv, VectorizedEnv


class _ObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = n


class _ContinuousActionSpace:
    def __init__(self, shape):
        self.shape = shape


class _FakeSingleEnv:
    observation_space = _ObservationSpace((4,))
    action_space = _DiscreteActionSpace(3)

    def reset(self, *args, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )

    def close(self):
        pass


class _FakeVectorizedEnv(VectorizedEnv):
    def __init__(self, worker_num=3):
        self.worker_num = worker_num
        self.env_info = {
            "observation_space": _ObservationSpace((5,)),
            "action_space": _ContinuousActionSpace((2,)),
            "env_type": "fake_vector",
            "env_id": "FakeVector-v0",
        }

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return np.zeros((self.worker_num, 5), dtype=np.float32)

    def step(self, action):
        pass

    def get_result(self):
        return None

    def close(self):
        pass


class _BrokenVectorizedEnv(_FakeVectorizedEnv):
    def __init__(self, worker_num=3):
        super().__init__(worker_num=worker_num)
        del self.env_info["action_space"]


def test_env_builder_package_imports_use_core_protocols_without_common_shims():
    adapter_module = importlib.import_module("env_builder.env_builder")
    adapter_seeding = importlib.import_module("env_builder.seeding")
    core_seeding = importlib.import_module("jax_baselines.core.seeding")
    adapter_atari = importlib.import_module("env_builder.atari_wrappers")

    assert adapter_seeding.seed_prngs is core_seeding.seed_prngs
    assert adapter_seeding.key_gen is core_seeding.key_gen
    assert adapter_seeding.set_global_seeds is core_seeding.set_global_seeds
    assert hasattr(adapter_seeding, "seed_env")
    assert hasattr(adapter_atari, "make_wrap_atari")
    assert adapter_module.VectorizedEnv is VectorizedEnv
    assert adapter_module.Env is Env
    assert adapter_module.EnvInfo is EnvInfo
    assert hasattr(importlib.import_module("env_builder"), "get_env_builder")


def test_single_env_protocol_accepts_structural_envs():
    env = _FakeSingleEnv()

    assert isinstance(env, SingleEnv)


def test_get_local_env_info_normalizes_single_env_without_concrete_backend_import():
    calls = []

    def _builder(worker, seed=None):
        calls.append((worker, seed))
        return _FakeSingleEnv()

    _, _, observation_space, action_size, worker_size, env_type = get_local_env_info(
        _builder,
        num_workers=1,
        seed=7,
    )

    assert calls == [(1, 7), (1, 8)]
    assert observation_space == [[4]]
    assert action_size == [3]
    assert worker_size == 1
    assert env_type == "SingleEnv"


def test_get_local_env_info_normalizes_vectorized_env_contract():
    calls = []

    def _builder(worker, seed=None):
        calls.append((worker, seed))
        return _FakeVectorizedEnv(worker_num=worker)

    _, _, observation_space, action_size, worker_size, env_type = get_local_env_info(
        _builder,
        num_workers=3,
        seed=11,
    )

    assert calls == [(3, 11), (1, 12)]
    assert observation_space == [[5]]
    assert action_size == [2]
    assert worker_size == 3
    assert env_type == "VectorizedEnv"


def test_get_local_env_info_requires_explicit_vectorized_env_info_contract():
    def _builder(worker, seed=None):
        return _BrokenVectorizedEnv(worker_num=worker)

    try:
        get_local_env_info(_builder, num_workers=2)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing env_info keys to fail")

    assert "VectorizedEnv env_info missing required keys: action_space" in message


def test_infer_action_meta_uses_structural_space_contract():
    discrete_type, discrete_conv = infer_action_meta(_DiscreteActionSpace(4))
    continuous_type, continuous_conv = infer_action_meta(_ContinuousActionSpace((2,)))

    assert discrete_type == "discrete"
    assert discrete_conv([2]) == 2
    assert continuous_type == "continuous"
    np.testing.assert_allclose(continuous_conv(np.array([6.0, -6.0])), np.array([1.0, -1.0]))
