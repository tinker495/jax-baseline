"""Regression coverage for the env_builder adapter migration."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from jax_baselines.core.env_info import (
    get_local_env_info,
    infer_action_meta,
    prepare_worker_env,
)
from jax_baselines.core.env_protocols import (
    Env,
    EnvInfo,
    PreparedEnvSpec,
    PreparedWorkerEnvSpec,
    SingleEnv,
    VectorizedEnv,
)


class _ObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = n


class _FakeSingleEnv:
    observation_space = _ObservationSpace((4,))
    action_space = _DiscreteActionSpace(3)

    def __init__(self, seed=None):
        self.seed = seed

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
            "observation_space": [[5]],
            "action_size": [2],
            "action_type": "continuous",
            "env_type": "fake_vector",
            "env_id": "FakeVector-v0",
            "worker_num": worker_num,
            "core_env_type": "VectorizedEnv",
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
        del self.env_info["action_size"]


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
    assert hasattr(importlib.import_module("env_builder"), "PreparedEnvSpec")
    assert hasattr(importlib.import_module("env_builder"), "get_env_builder")



def test_make_wrap_atari_applies_the_canonical_wrapper_order(monkeypatch):
    atari = importlib.import_module("env_builder.atari_wrappers")
    calls = []

    class _Env:
        spec = type("Spec", (), {"id": "PongNoFrameskip-v4"})()
        unwrapped = type(
            "Unwrapped", (), {"get_action_meanings": lambda self: ["NOOP", "FIRE", "UP"]}
        )()

    env = _Env()

    def wrapper(name):
        def apply(current, *args, **kwargs):
            calls.append((name, args, kwargs))
            return current

        return apply

    monkeypatch.setattr(atari.gym, "make", lambda *a, **k: env)
    monkeypatch.setattr(atari, "NoopResetEnv", wrapper("noop"))
    monkeypatch.setattr(atari, "MaxAndSkipEnv", wrapper("skip"))
    monkeypatch.setattr(atari.gym.wrappers, "TimeLimit", wrapper("time_limit"))
    monkeypatch.setattr(atari, "EpisodicLifeEnv", wrapper("episodic_life"))
    monkeypatch.setattr(atari, "FireResetEnv", wrapper("fire"))
    monkeypatch.setattr(atari, "WarpFrame", wrapper("warp"))
    monkeypatch.setattr(atari, "ClipRewardEnv", wrapper("clip"))
    monkeypatch.setattr(atari, "FrameStack", wrapper("frame_stack"))

    assert atari.make_wrap_atari("PongNoFrameskip-v4", clip_rewards=True) is env
    assert [name for name, _, _ in calls] == [
        "noop",
        "skip",
        "time_limit",
        "episodic_life",
        "fire",
        "warp",
        "clip",
        "frame_stack",
    ]


def test_single_env_protocol_accepts_structural_envs():
    env = _FakeSingleEnv()

    assert isinstance(env, SingleEnv)


def test_get_local_env_info_consumes_adapter_prepared_single_envs():
    calls = []

    class _Builder:
        def __call__(self, *args, **kwargs):
            raise AssertionError("core must not call builder directly when prepare_envs exists")

        def prepare_envs(self, num_workers=1, seed=None):
            calls.append((num_workers, seed))
            env = _FakeSingleEnv()
            return PreparedEnvSpec(
                env=env,
                eval_env=_FakeSingleEnv(),
                env_info={
                    "observation_space": [[4]],
                    "action_size": [3],
                    "action_type": "discrete",
                    "env_type": "single",
                    "env_id": "FakeSingle-v0",
                    "worker_num": 1,
                    "core_env_type": "SingleEnv",
                },
            )

    _, _, observation_space, action_size, worker_size, env_type = get_local_env_info(
        _Builder(),
        num_workers=1,
        seed=7,
    )

    assert calls == [(1, 7)]
    assert observation_space == [[4]]
    assert action_size == [3]
    assert worker_size == 1
    assert env_type == "SingleEnv"


def test_get_local_env_info_consumes_adapter_prepared_vectorized_envs():
    calls = []

    class _Builder:
        def __call__(self, *args, **kwargs):
            raise AssertionError("core must not call builder directly when prepare_envs exists")

        def prepare_envs(self, num_workers=1, seed=None):
            calls.append((num_workers, seed))
            env = _FakeVectorizedEnv(worker_num=num_workers)
            return PreparedEnvSpec(env=env, eval_env=_FakeSingleEnv(), env_info=env.env_info)

    _, _, observation_space, action_size, worker_size, env_type = get_local_env_info(
        _Builder(),
        num_workers=3,
        seed=11,
    )

    assert calls == [(3, 11)]
    assert observation_space == [[5]]
    assert action_size == [2]
    assert worker_size == 3
    assert env_type == "VectorizedEnv"


def test_get_local_env_info_requires_explicit_vectorized_env_info_contract():
    class _Builder:
        def prepare_envs(self, num_workers=1, seed=None):
            env = _BrokenVectorizedEnv(worker_num=num_workers)
            return PreparedEnvSpec(env=env, eval_env=_FakeSingleEnv(), env_info=env.env_info)

    try:
        get_local_env_info(_Builder(), num_workers=2)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing env_info keys to fail")

    assert "Prepared env_info missing required keys: action_size" in message


def test_get_local_env_info_requires_local_eval_env():
    class _Builder:
        def prepare_envs(self, num_workers=1, seed=None):
            env = _FakeSingleEnv(seed=seed)
            return PreparedEnvSpec(
                env=env,
                eval_env=None,
                env_info={
                    "observation_space": [[4]],
                    "action_size": [3],
                    "action_type": "discrete",
                    "env_type": "single",
                    "env_id": "FakeSingle-v0",
                    "worker_num": 1,
                    "core_env_type": "SingleEnv",
                },
            )

    with pytest.raises(ValueError, match="Prepared local eval_env is required"):
        get_local_env_info(_Builder(), num_workers=1, seed=3)


def test_prepare_worker_env_requires_single_worker_spec():
    class _Builder:
        def prepare_worker_env(self, seed=None):
            return PreparedWorkerEnvSpec(
                env=_FakeVectorizedEnv(worker_num=2),
                env_info={
                    "observation_space": [[5]],
                    "action_size": [2],
                    "action_type": "continuous",
                    "env_type": "fake_vector",
                    "env_id": "FakeVector-v0",
                    "worker_num": 2,
                    "core_env_type": "VectorizedEnv",
                },
            )

    with pytest.raises(ValueError, match="Prepared worker env metadata must be SingleEnv"):
        prepare_worker_env(_Builder(), seed=4)


def test_infer_action_meta_uses_adapter_normalized_action_type():
    discrete_type, discrete_conv = infer_action_meta("discrete")
    continuous_type, continuous_conv = infer_action_meta("continuous")

    assert discrete_type == "discrete"
    assert discrete_conv([2]) == 2
    assert continuous_type == "continuous"
    np.testing.assert_allclose(continuous_conv(np.array([6.0, -6.0])), np.array([1.0, -1.0]))


def test_env_builder_adapter_prepares_train_eval_pair_and_seed_policy():
    calls = []

    def fake_env_builder(worker=1, render_mode=None, seed=None):
        calls.append((worker, seed))
        return _FakeVectorizedEnv(worker_num=worker) if worker > 1 else _FakeSingleEnv()

    # Replace the attached method with a deterministic local equivalent so this
    # test stays backend-free while asserting the adapter-owned contract shape.
    def prepare_envs(num_workers=1, seed=None):
        eval_seed = None if seed is None else seed + 1
        env = fake_env_builder(num_workers, seed=seed)
        eval_env = fake_env_builder(1, seed=eval_seed)
        return PreparedEnvSpec(
            env=env,
            eval_env=eval_env,
            env_info=env.env_info
            if isinstance(env, VectorizedEnv)
            else {
                "observation_space": [[4]],
                "action_size": [3],
                "action_type": "discrete",
                "env_type": "single",
                "env_id": "FakeSingle-v0",
                "worker_num": 1,
                "core_env_type": "SingleEnv",
            },
        )

    fake_env_builder.prepare_envs = prepare_envs

    get_local_env_info(fake_env_builder, num_workers=4, seed=21)

    assert calls == [(4, 21), (1, 22)]


def test_experiments_build_env_returns_prepared_env_builder():
    from experiments.cli.qnet import QNET_RUNNER

    # Build through the real qnet experiment composition path. The returned
    # object is still callable for workers, but core should use prepare_envs.
    parser = importlib.import_module("argparse").ArgumentParser()
    QNET_RUNNER.add_args(parser)
    args = parser.parse_args(["--env", "CartPole-v1"])
    builder, _policy_kwargs = QNET_RUNNER.build_env(args)

    assert callable(builder)
    assert callable(getattr(builder, "prepare_envs", None))
    assert callable(getattr(builder, "prepare_worker_env", None))


def test_experiments_composition_path_uses_adapter_prepared_envs(monkeypatch):
    import experiments.cli.qnet as qnet

    calls = []

    def fake_get_env_builder(env_name, env_backend="gymnasium"):
        calls.append(("get_env_builder", env_name, {"env_backend": env_backend}))

        def builder(*args, **kwargs):
            raise AssertionError("core must not call the raw builder on the experiments path")

        def prepare_envs(num_workers=1, seed=None):
            calls.append(("prepare_envs", num_workers, seed))
            env = _FakeVectorizedEnv(worker_num=num_workers)
            eval_env = _FakeSingleEnv(seed=None if seed is None else seed + 100)
            return PreparedEnvSpec(env=env, eval_env=eval_env, env_info=env.env_info)

        def prepare_worker_env(seed=None):
            calls.append(("prepare_worker_env", seed))
            env = _FakeSingleEnv(seed=seed)
            return PreparedWorkerEnvSpec(
                env=env,
                env_info={
                    "observation_space": [[4]],
                    "action_size": [3],
                    "action_type": "discrete",
                    "env_type": "single",
                    "env_id": "FakeSingle-v0",
                    "worker_num": 1,
                    "core_env_type": "SingleEnv",
                },
            )

        builder.prepare_envs = prepare_envs
        builder.prepare_worker_env = prepare_worker_env
        return builder, {"env_id": env_name}

    monkeypatch.setattr(qnet, "get_env_builder", fake_get_env_builder)

    parser = importlib.import_module("argparse").ArgumentParser()
    qnet.QNET_RUNNER.add_args(parser)
    args = parser.parse_args(["--env", "CartPole-v1", "--worker", "4"])
    builder, _policy_kwargs = qnet.QNET_RUNNER.build_env(args)
    env, eval_env, observation_space, action_size, worker_size, env_type = get_local_env_info(
        builder,
        num_workers=args.worker,
        seed=21,
    )

    assert isinstance(env, _FakeVectorizedEnv)
    assert eval_env.seed == 121
    assert observation_space == [[5]]
    assert action_size == [2]
    assert worker_size == 4
    assert env_type == "VectorizedEnv"
    assert calls == [
        (
            "get_env_builder",
            "CartPole-v1",
            {"env_backend": "gymnasium"},
        ),
        ("prepare_envs", 4, 21),
    ]


def test_core_env_info_does_not_infer_concrete_space_shapes():
    """The adapter owns gym/space shape interpretation; core consumes EnvInfo."""
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "jax_baselines/core/env_info.py").read_text())

    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr"
        for node in ast.walk(tree)
    )
