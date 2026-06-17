from __future__ import annotations

import glob
import os
from typing import Any, Optional

from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
from tqdm.auto import trange

from jax_baselines.core.eval import run_test_episodes
from jax_baselines.core.hparams import add_hparams


def _get_latest_run_id(local_dir, experiment_name, run_name):
    """Return the latest numbered TensorBoard run id for a run name."""

    max_run_id = 0
    for path in glob.glob(f"{local_dir}/{experiment_name}/{run_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if run_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit():
            max_run_id = max(max_run_id, int(ext))
    return max_run_id


class TensorboardRun:
    def __init__(self, dir: str):
        self.dir = dir
        self._writer = SummaryWriter(dir)

    def log_param(self, hparam_dict):
        exp, ssi, sei = hparams(hparam_dict, {})

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
                for leaf in ("episode_reward", "original_reward", "episode_length", "timeout_rate")
            },
        }
        self._writer.add_custom_scalars(layout)

    def get_local_path(self, path):
        return os.path.join(self.dir, path)


class TensorboardLogger:
    def __init__(self, run_name: str, experiment_name: str, local_dir: str, agent: Optional[Any]):
        self.run_name = run_name
        self.local_dir = os.path.join(
            local_dir,
            experiment_name,
            f"{run_name}_{_get_latest_run_id(local_dir, experiment_name, run_name) + 1:02d}",
        )
        self.run = TensorboardRun(self.local_dir)
        if agent is not None:
            try:
                self.log_hparams(agent)
            except Exception:
                # Preserve historical permissiveness: hparam logging must not break training.
                pass

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


def record_and_test(env_builder, logger_run, actions_eval_fn, episode, conv_action=None):
    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

    directory = logger_run.get_local_path("video")
    os.makedirs(directory, exist_ok=True)
    test_env = env_builder(1, render_mode="rgb_array")
    render_env = RecordVideo(test_env, directory, episode_trigger=lambda x: True)
    render_env = RecordEpisodeStatistics(render_env)
    with render_env:
        return run_test_episodes(render_env, actions_eval_fn, episode, conv_action)
