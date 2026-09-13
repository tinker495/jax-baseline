from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

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
    metadata: ClassVar[dict[str, int]] = {}

    def __init__(self):
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1
        return {"unified_obs": np.array([0.0])}, {}

    def step(self, action):
        return {"unified_obs": np.array([0.0])}, 3.0, True, False, {}

    def close(self):
        self.closed = True

    def render(self):
        return np.zeros((16, 32, 3), dtype=np.uint8).transpose(1, 0, 2)


def test_experiments_record_and_test_writes_videos(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    built = []
    env = _OneStepEnv()

    def env_builder(worker_size, render_mode=None):
        built.append((worker_size, render_mode))
        return env

    avg, std = runtime_adapters.record_and_test(
        env_builder,
        _LoggerRun(tmp_path),
        actions_eval_fn=lambda obs: np.array([0]),
        episode=2,
    )

    assert built == [(1, "rgb_array")]
    videos = sorted((tmp_path / "video").glob("*.mp4"))
    assert [imageio_ffmpeg.count_frames_and_secs(video)[0] for video in videos] == [2, 2]
    assert env.closed
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
                    "observation_space": {"unified_obs": [1]},
                    "action_size": [2],
                    "action_type": "discrete",
                    "env_type": "single",
                    "env_id": "Fake-v0",
                    "worker_num": 1,
                    "core_env_type": "SingleEnv",
                    "runtime": {
                        "backend": "fake",
                        "backend_env_id": "Fake-v0",
                        "seed": seed,
                        "seed_rule": "constructor seed",
                        "reward_clipping": "none",
                        "episodic_life": False,
                    },
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
