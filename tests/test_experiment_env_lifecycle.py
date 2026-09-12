from dataclasses import replace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

import experiments.cli._run as run_mod
import experiments.runtime_adapters as adapters
from env_builder.mjlab_env import MjlabSingleEnv
from experiments.cli.dpg import DPG_RUNNER
from jax_baselines.core.env_protocols import EnvironmentMetadata


class _Env(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Dict(
        {"unified_obs": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)}
    )
    runtime: ClassVar[EnvironmentMetadata] = {
        "backend": "fake",
        "backend_env_id": "Fake-v0",
        "seed": 0,
        "seed_rule": "constructor seed",
        "reward_clipping": "none",
        "episodic_life": False,
    }

    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("close")


class _MjlabEnv(MjlabSingleEnv):
    def __init__(self, steps=2):
        self.metadata = {"render_fps": 10}
        self.steps = steps
        self.frame = 0
        self.closed = False
        self.log_calls = []

    def reset(self):
        self.frame = 0
        return {"unified_obs": np.array([0.0])}, {}

    def step(self, action):
        del action
        self.frame += 1
        return {"unified_obs": np.array([0.0])}, 1.0, self.frame == self.steps, False, {}

    def render(self):
        return np.full((16, 16, 3), self.frame, dtype=np.uint8)

    def log_metrics(self, logger, steps, *, namespace="rollout", active=None, flush=True):
        self.log_calls.append((logger, namespace, flush))

    def close(self):
        self.closed = True


def test_headless_test_never_requests_rendering_and_always_closes(monkeypatch):
    calls = []
    env = _Env(calls)

    def builder(worker, **kwargs):
        calls.append((worker, kwargs))
        return env

    monkeypatch.setattr(
        adapters, "run_test_episodes", lambda *args, **kwargs: calls.append("run") or 3
    )
    assert adapters.headless_test(builder, None, object(), 2) == 3
    assert calls == [(1, {}), "run", "close"]

    monkeypatch.setattr(
        adapters, "run_test_episodes", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError())
    )
    with pytest.raises(ValueError):
        adapters.headless_test(builder, None, object(), 2)
    assert calls[-1] == "close"


def test_record_and_test_writes_one_mjlab_video_per_episode(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

    env = _MjlabEnv()
    logger = type("Logger", (), {"get_local_path": lambda self, path: str(tmp_path / path)})()

    assert adapters.record_and_test(
        lambda worker, render_mode=None: env,
        logger,
        lambda obs: np.array([0.0]),
        episode=2,
    ) == (2.0, 0.0)

    videos = sorted((tmp_path / "video").glob("rl-video-episode-*.mp4"))
    assert [video.name for video in videos] == [
        "rl-video-episode-0.mp4",
        "rl-video-episode-1.mp4",
    ]
    assert [imageio_ffmpeg.count_frames_and_secs(video)[0] for video in videos] == [3, 3]
    assert env.log_calls == [(logger, "test", False)] * 4 + [(logger, "test", True)]
    assert env.closed


def test_record_and_test_closes_mjlab_env_when_evaluation_fails(monkeypatch, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    env = _MjlabEnv()
    logger = type("Logger", (), {"get_local_path": lambda self, path: str(tmp_path / path)})()

    def fail_after_reset(test_env, *args, **kwargs):
        test_env.reset()
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(adapters, "run_test_episodes", fail_after_reset)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        adapters.record_and_test(lambda worker, render_mode=None: env, logger, object(), episode=1)

    assert env.closed


@pytest.mark.parametrize(
    ("supports_render", "record_test_fn"),
    [(False, adapters.headless_test), (True, adapters.record_and_test)],
)
@pytest.mark.parametrize("failure", [None, "learn", "test"])
def test_run_family_closes_distinct_envs_and_selects_headless_callback(
    monkeypatch, failure, supports_render, record_test_fn
):
    events = []
    train = _Env(events)
    evaluation = _Env(events)

    class Builder:
        pass

    Builder.supports_render = supports_render

    class Agent:
        def __init__(self, *args, **kwargs):
            self.env = train
            self.eval_env = evaluation
            self.policy_kwargs = kwargs["policy_kwargs"]

        def learn(self, *args, record_test_fn=None, **kwargs):
            events.append(("learn", record_test_fn))
            if failure == "learn":
                raise RuntimeError("learn")

        def test(self):
            events.append("test")
            if failure == "test":
                raise RuntimeError("test")

    spec = replace(DPG_RUNNER.algos["DDPG"], cls=Agent)
    runner = replace(
        DPG_RUNNER,
        algos={**DPG_RUNNER.algos, "DDPG": spec},
        build_env=lambda args: (Builder(), {}),
    )
    monkeypatch.setattr(run_mod, "resolve_maker", lambda *args: object())
    monkeypatch.setattr(run_mod, "collect_run_metadata", lambda *args, **kwargs: {})

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run_mod.run_family(runner, ["--algo", "DDPG", "--steps", "1"])
    else:
        run_mod.run_family(runner, ["--algo", "DDPG", "--steps", "1"])

    assert events[0] == ("learn", record_test_fn)
    assert events.count("close") == 2
    if failure != "learn":
        assert events.index("test") > events.index("close")


def test_run_family_deduplicates_shared_train_eval_identity(monkeypatch):
    events = []
    env = _Env(events)

    class Agent:
        def __init__(self, *args, **kwargs):
            self.env = self.eval_env = env
            self.policy_kwargs = kwargs["policy_kwargs"]

        def learn(self, *args, **kwargs):
            pass

        def test(self):
            events.append("test")

    spec = replace(DPG_RUNNER.algos["DDPG"], cls=Agent)
    runner = replace(
        DPG_RUNNER,
        algos={**DPG_RUNNER.algos, "DDPG": spec},
        build_env=lambda args: (lambda: None, {}),
    )
    monkeypatch.setattr(run_mod, "resolve_maker", lambda *args: object())
    monkeypatch.setattr(run_mod, "collect_run_metadata", lambda *args, **kwargs: {})

    run_mod.run_family(runner, ["--algo", "DDPG", "--steps", "1"])

    assert events == ["close", "test"]


def test_close_agent_envs_attempts_both_after_close_failure():
    events = []

    class BrokenEnv:
        def close(self):
            events.append("broken")
            raise RuntimeError("close failed")

    agent = type("Agent", (), {"env": BrokenEnv(), "eval_env": _Env(events)})()

    with pytest.raises(RuntimeError, match="close failed"):
        run_mod._close_agent_envs(agent)

    assert events == ["broken", "close"]
