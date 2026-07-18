from __future__ import annotations

import numpy as np

from experiments import runtime_adapters
from jax_baselines.core.env_protocols import PreparedWorkerEnvSpec


class _LoggerRun:
    def __init__(self, root):
        self.root = root

    def get_local_path(self, path):
        return str(self.root / path)


class _OneStepEnv:
    observation_space = object()
    action_space = object()

    def __init__(self):
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1
        return {"obs": np.array([0.0])}, {}

    def step(self, action):
        return {"obs": np.array([0.0])}, 3.0, True, False, {}

    def close(self):
        self.closed = True


class _Wrapper:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        return False

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


def test_experiments_record_and_test_uses_video_wrappers(monkeypatch, tmp_path):
    calls = []

    class FakeRecordVideo(_Wrapper):
        def __init__(self, env, directory, episode_trigger):
            calls.append(("video", directory, episode_trigger(0)))
            super().__init__(env, directory, episode_trigger=episode_trigger)

    class FakeRecordEpisodeStatistics(_Wrapper):
        def __init__(self, env):
            calls.append(("stats",))
            super().__init__(env)

    import gymnasium.wrappers

    monkeypatch.setattr(gymnasium.wrappers, "RecordVideo", FakeRecordVideo)
    monkeypatch.setattr(gymnasium.wrappers, "RecordEpisodeStatistics", FakeRecordEpisodeStatistics)

    built = []

    def env_builder(worker_size, render_mode=None):
        built.append((worker_size, render_mode))
        return _OneStepEnv()

    avg, std = runtime_adapters.record_and_test(
        env_builder,
        _LoggerRun(tmp_path),
        actions_eval_fn=lambda obs: np.array([0]),
        episode=2,
    )

    assert built == [(1, "rgb_array")]
    assert calls == [("video", str(tmp_path / "video"), True), ("stats",)]
    assert avg == 3.0
    assert std == 0.0


def test_core_record_and_test_uses_worker_env_protocol(tmp_path):
    from jax_baselines.core.eval import record_and_test

    env = _OneStepEnv()
    calls = []

    class Builder:
        def prepare_worker_env(self, seed=None):
            calls.append(seed)
            return PreparedWorkerEnvSpec(
                env=env,
                env_info={
                    "observation_space": {"obs": [1]},
                    "action_size": [2],
                    "action_type": "discrete",
                    "env_type": "single",
                    "env_id": "Fake-v0",
                    "worker_num": 1,
                    "core_env_type": "SingleEnv",
                },
            )

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("core evaluation must use prepare_worker_env")

    avg, std = record_and_test(
        Builder(),
        _LoggerRun(tmp_path),
        actions_eval_fn=lambda obs: np.array([0]),
        episode=2,
    )

    assert calls == [None]
    assert env.closed is True
    assert avg == 3.0
    assert std == 0.0
