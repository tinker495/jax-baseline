from __future__ import annotations

import os
from contextlib import ExitStack, closing, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
from tqdm.auto import trange

from experiments.run_metadata import write_run_metadata
from jax_baselines.core.env_protocols import EvaluationContextEnv, VectorizedEvalEnv
from jax_baselines.core.eval import run_test_episodes
from jax_baselines.core.hparams import add_hparams


def create_run_directory(local_dir: str, experiment_name: str, run_name: str) -> str:
    """Reserve a numbered directory without sharing artifacts with concurrent runs."""

    max_run_id = 0
    for path in (Path(local_dir) / experiment_name).glob(f"{run_name}_[0-9]*"):
        prefix, _, ext = path.name.rpartition("_")
        if prefix == run_name and ext.isdigit():
            max_run_id = max(max_run_id, int(ext))
    while True:
        max_run_id += 1
        directory = Path(local_dir) / experiment_name / f"{run_name}_{max_run_id:02d}"
        try:
            directory.mkdir(parents=True)
        except FileExistsError:
            continue
        return str(directory)


class TensorboardRun:
    def __init__(self, dir: str, extra_hparams: dict[str, str] | None = None):
        self.dir = dir
        self._writer = SummaryWriter(dir)
        self._extra_hparams = {} if extra_hparams is None else dict(extra_hparams)

    def log_param(self, hparam_dict):
        exp, ssi, sei = hparams({**hparam_dict, **self._extra_hparams}, {})

        self._writer.file_writer.add_summary(exp)
        self._writer.file_writer.add_summary(ssi)
        self._writer.file_writer.add_summary(sei)

    def log_metric(self, key, value, step=None):
        self._writer.add_scalar(key, value, step)

    def log_histogram(self, key, value, step=None):
        self._writer.add_histogram(key, value, step)

    def declare_multiline_layout(self, eps):
        """TensorBoard custom-scalars layout overlaying the per-epsilon rollout
        curves (distributed APE-X). TensorBoard-only; other backends no-op."""
        layout = {
            "rollout": {
                leaf: [
                    "Multiline",
                    [f"rollout/{leaf}/eps{e:.2f}" for e in eps] + [f"rollout/{leaf}"],
                ]
                for leaf in (
                    "episode_reward",
                    "original_reward",
                    "episode_length",
                    "timeout_rate",
                )
            },
        }
        self._writer.add_custom_scalars(layout)

    def get_local_path(self, path):
        return os.path.join(self.dir, path)


class TensorboardLogger:
    def __init__(
        self,
        run_name: str,
        experiment_name: str,
        local_dir: str,
        agent: Any | None,
        *,
        extra_hparams: dict[str, str] | None = None,
        run_metadata: dict[str, object] | None = None,
    ):
        self.run_name = run_name
        self.local_dir = create_run_directory(local_dir, experiment_name, run_name)
        self.run = TensorboardRun(self.local_dir, extra_hparams)
        write_run_metadata(self.local_dir, run_metadata)
        if agent is not None:
            self.log_hparams(agent)

    def log_hparams(self, agent_or_hparams):
        if agent_or_hparams is None:
            return
        if isinstance(agent_or_hparams, dict):
            self.run.log_param(agent_or_hparams)
        else:
            add_hparams(agent_or_hparams, self.run)

    def __enter__(self):
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def close(self):
        self.run._writer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_progress(*args, **kwargs):
    return trange(*args, **kwargs)


class _VideoRecorder:
    def __init__(self, env, directory):
        from imageio_ffmpeg import write_frames

        self.env = env
        self.directory = Path(directory)
        self.fps = {"render_fps": 30, **env.metadata}["render_fps"]
        self._write_frames = write_frames
        self._writer = None
        self._episode = 0

    def _close_writer(self):
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()

    def _write_frame(self):
        frame = np.ascontiguousarray(self.env.render())
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = self._write_frames(
                self.directory / f"rl-video-episode-{self._episode}.mp4",
                (width, height),
                fps=self.fps,
            )
            self._episode += 1
            self._writer.send(None)
        self._writer.send(frame)

    def reset(self):
        self._close_writer()
        result = self.env.reset()
        self._write_frame()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._write_frame()
        if result[2] or result[3]:
            self._close_writer()
        return result

    def close(self):
        self._close_writer()


class _VectorVideoRecorder(_VideoRecorder, VectorizedEvalEnv):
    def get_info(self):
        return self.env.get_info()

    def current_obs(self):
        return self.env.current_obs()

    def step(self, action):
        self.env.step(action)

    def get_result(self):
        result = self.env.get_result()
        self._write_frame()
        if result[2][0] or result[3][0]:
            self._close_writer()
        return result


def record_and_test(
    env_builder,
    logger_run,
    actions_eval_fn,
    episode,
    conv_action=None,
    *,
    existing_env=None,
    training_env=None,
):
    import gymnasium as gym

    from env_builder.env_builder import EnvPoolVectorizedEnv, GymVectorizedEnv
    from env_builder.gym_rendering import GymRenderingWrapper
    from env_builder.mjlab_env import MjlabSingleEnv, MjlabVectorizedEnv

    if episode < 1:
        raise ValueError("episode must be positive")
    if isinstance(training_env, (MjlabSingleEnv, MjlabVectorizedEnv)):
        existing_env = training_env
    directory = Path(logger_run.get_local_path("video"))
    directory.mkdir(parents=True, exist_ok=True)
    rendering_env = existing_env
    while isinstance(rendering_env, gym.Wrapper) and not isinstance(
        rendering_env, GymRenderingWrapper
    ):
        rendering_env = rendering_env.env
    with ExitStack() as stack:
        if isinstance(existing_env, (MjlabSingleEnv, MjlabVectorizedEnv)):
            test_env = existing_env
            stack.enter_context(test_env.testing_context())
            stack.enter_context(test_env.rendering_context())
        elif (
            isinstance(rendering_env, (GymRenderingWrapper, GymVectorizedEnv))
            and rendering_env.supports_rendering
        ):
            test_env = existing_env
            if isinstance(test_env, EvaluationContextEnv):
                stack.enter_context(test_env.evaluation_context())
            stack.enter_context(rendering_env.rendering_context())
        elif (
            isinstance(existing_env, (gym.Env, EnvPoolVectorizedEnv))
            and existing_env.render_mode == "rgb_array"
        ):
            test_env = existing_env
            if isinstance(test_env, EvaluationContextEnv):
                stack.enter_context(test_env.evaluation_context())
        else:
            test_env = stack.enter_context(closing(env_builder(1, render_mode="rgb_array")))
        print(
            f"Video environment: {'reused' if test_env is existing_env else 'created'}; "
            f"recording worker 0 to {directory}"
        )
        recorder = (
            _VectorVideoRecorder if isinstance(test_env, VectorizedEvalEnv) else _VideoRecorder
        )
        render_env = stack.enter_context(closing(recorder(test_env, directory)))
        return run_test_episodes(
            render_env,
            actions_eval_fn,
            episode,
            conv_action,
            logger_run=logger_run,
            logging_env=test_env,
        )


def headless_test(
    env_builder,
    logger_run,
    actions_eval_fn,
    episode,
    conv_action=None,
    *,
    existing_env=None,
    training_env=None,
):
    from env_builder.mjlab_env import MjlabSingleEnv, MjlabVectorizedEnv

    if isinstance(training_env, (MjlabSingleEnv, MjlabVectorizedEnv)):
        existing_env = training_env
    with (
        closing(env_builder(1)) if existing_env is None else nullcontext(existing_env) as test_env,
        (
            test_env.testing_context()
            if isinstance(test_env, (MjlabSingleEnv, MjlabVectorizedEnv))
            else test_env.evaluation_context()
            if existing_env is not None and isinstance(test_env, EvaluationContextEnv)
            else nullcontext()
        ),
    ):
        return run_test_episodes(
            test_env, actions_eval_fn, episode, conv_action, logger_run=logger_run
        )
