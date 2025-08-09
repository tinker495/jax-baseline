import numpy as np
import ray

from jax_baselines.common.logger import TensorboardLogger


@ray.remote
class Param_server(object):
    def __init__(self, params) -> None:
        self.params = params

    def get_params(self):
        return self.params

    def update_params(self, params):
        self.params = params


@ray.remote
class Logger_server(object):
    def __init__(self, log_dir, log_name) -> None:
        # pass None to avoid attempting to extract hparams from this Ray actor
        # (which would serialize the whole actor). Hyperparameters should be
        # registered explicitly via `register_hparams` with a plain dict.
        self.writer = TensorboardLogger(log_name, "experiment", log_dir, None)
        self.step = 0
        self.old_step = 0
        self.save_dict = dict()
        with self.writer as (summary, save_path):
            self.save_path = save_path

    def get_log_dir(self):
        return self.save_path

    def add_multiline(self, eps):
        with self.writer as (summary, _):
            layout = {
                "env": {
                    "episode_reward": [
                        "Multiline",
                        [f"env/episode_reward/eps{e:.2f}" for e in eps] + ["env/episode_reward"],
                    ],
                    "original_reward": [
                        "Multiline",
                        [f"env/original_reward/eps{e:.2f}" for e in eps] + ["env/original_reward"],
                    ],
                    "episode_len": [
                        "Multiline",
                        [f"env/episode_len/eps{e:.2f}" for e in eps] + ["env/episode_len"],
                    ],
                    "time_over": [
                        "Multiline",
                        [f"env/time_over/eps{e:.2f}" for e in eps] + ["env/time_over"],
                    ],
                },
            }
            summary.add_custom_scalars(layout)

    def log_trainer(self, step, log_dict):
        self.step = step
        with self.writer as (summary, _):
            for key, value in log_dict.items():
                summary.add_scalar(key, value, self.step)

    def log_worker(self, log_dict, episode):
        if self.old_step != self.step:
            with self.writer as (summary, _):
                for key, value in self.save_dict.items():
                    summary.add_scalar(key, np.mean(value), self.step)
                self.save_dict = dict()
                self.old_step = self.step
        for key, value in log_dict.items():
            if key in self.save_dict:
                self.save_dict[key].append(value)
            else:
                self.save_dict[key] = [value]

    def last_update(self):
        with self.writer as (summary, _):
            for key, value in self.save_dict.items():
                summary.add_scalar(key, np.mean(value), self.step)

    def register_hparams(self, hparams: dict):
        """Register hyperparameters (plain dict) to be logged to TensorBoard.

        This should be called from the trainer/process that has the real agent
        object, using `get_hyper_params(agent)` to build a serializable dict,
        and then `logger.register_hparams.remote(hparams)` to send it to this
        Ray actor.
        """
        try:
            self.writer.log_hparams(hparams)
        except Exception:
            # silently ignore logging errors
            pass
