import os

import numpy as np

from jax_baselines.core.runtime_adapters import NoOpLogger


def _ray():
    from importlib import import_module

    return import_module("ray")


class _RayActor:
    @classmethod
    def remote(cls, *args, **kwargs):
        return _ray().remote(cls).remote(*args, **kwargs)


class Param_server(_RayActor):
    def __init__(self, params) -> None:
        self.params = params

    def get_params(self):
        return self.params

    def update_params(self, params):
        self.params = params


class Logger_server(_RayActor):
    def __init__(
        self, log_dir, log_name, experiment_name="experiment", logger_factory=None
    ) -> None:
        # Pass None as the agent to avoid attempting to extract hparams from this Ray actor
        # (which would serialize the whole actor). Hyperparameters should be
        # registered explicitly via `register_hparams` with a plain dict.
        logger_factory = logger_factory or NoOpLogger
        self.logger = logger_factory(log_name, experiment_name, log_dir, None)
        self.step = 0
        self.old_step = 0
        self.save_dict = dict()
        with self.logger as run:
            self.save_path = os.path.normpath(run.get_local_path(""))

    def get_log_dir(self):
        return self.save_path

    def add_multiline(self, eps):
        with self.logger as run:
            run.declare_multiline_layout(eps)

    def log_trainer(self, step, log_dict):
        self.step = step
        with self.logger as run:
            for key, value in log_dict.items():
                run.log_metric(key, value, self.step)

    def log_worker(self, log_dict, episode):
        if self.old_step != self.step:
            with self.logger as run:
                for key, value in self.save_dict.items():
                    run.log_metric(key, np.mean(value), self.step)
                self.save_dict = dict()
                self.old_step = self.step
        for key, value in log_dict.items():
            if key in self.save_dict:
                self.save_dict[key].append(value)
            else:
                self.save_dict[key] = [value]

    def last_update(self):
        with self.logger as run:
            for key, value in self.save_dict.items():
                run.log_metric(key, np.mean(value), self.step)

    def close(self):
        """Finalize the run. The centralized logger actor outlives every
        ``with self.logger`` block, so the backend's run is closed here once the
        driver is done — not on per-call ``__exit__`` (which keeps it open) nor
        only on actor GC."""
        self.logger.close()

    def register_hparams(self, hparams: dict):
        """Register hyperparameters (plain dict) to be logged.

        This should be called from the trainer/process that has the real agent
        object, using `get_hyper_params(agent)` to build a serializable dict,
        and then `logger.register_hparams.remote(hparams)` to send it to this
        Ray actor.
        """
        self.logger.log_hparams(hparams)
