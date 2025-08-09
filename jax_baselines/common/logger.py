import glob
import os

from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from jax_baselines.common.utils import add_hparams


def _get_latest_run_id(local_dir, experiment_name, run_name):
    """returns the latest run number for the given log name and log path, by finding the greatest number in the
    directories.

    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob("{}/{}/{}_[0-9]*".format(local_dir, experiment_name, run_name)):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if (
            run_name == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id


class TensorboardRun:
    def __init__(self, dir: str):
        self.dir = dir
        self.writer = SummaryWriter(dir)

    def log_param(self, hparam_dict):
        exp, ssi, sei = hparams(hparam_dict, {})

        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

    def log_metric(self, key, value, step=None):
        self.writer.add_scalar(key, value, step)

    def log_histogram(self, key, value, step=None):
        self.writer.add_histogram(key, value, step)

    def get_local_path(self, path):
        return os.path.join(self.dir, path)


class TensorboardContext:
    """A context object that is both iterable (for tuple-unpacking) and
    exposes the TensorboardRun API (for direct attribute access).

    This allows existing call sites to use either:
      with TensorboardLogger(...) as (summary, save_path):
    or
      with TensorboardLogger(...) as logger_run:
    where `summary` is the underlying SummaryWriter and `logger_run` has
    methods like `log_metric` and `get_local_path`.
    """

    def __init__(self, run: TensorboardRun, local_dir: str):
        self.run = run
        self.local_dir = local_dir

    # Iterable protocol so callers can do: with logger as (summary, save_path):
    def __iter__(self):
        yield self.run.writer
        yield self.local_dir

    # Delegate common APIs to the underlying TensorboardRun
    def log_param(self, hparam_dict):
        return self.run.log_param(hparam_dict)

    def log_metric(self, key, value, step=None):
        return self.run.log_metric(key, value, step)

    def log_histogram(self, key, value, step=None):
        return self.run.log_histogram(key, value, step)

    def get_local_path(self, path):
        return self.run.get_local_path(path)

    @property
    def writer(self):
        return self.run.writer


class TensorboardLogger:
    def __init__(self, run_name: str, experiment_name: str, local_dir: str, agent: any):
        """
        Create a TensorBoard logger and optionally log hyperparameters from `agent`.

        If `agent` is None, hyperparameter logging is deferred and can be done later
        via `log_hparams`.
        """
        self.run_name = run_name
        self.local_dir = os.path.join(
            local_dir,
            experiment_name,
            f"{run_name}_{_get_latest_run_id(local_dir, experiment_name, run_name)+1:02d}",
        )
        self.run = TensorboardRun(self.local_dir)
        # allow deferred logging if agent is None
        if agent is not None:
            try:
                # if agent is a dict of hparams
                if isinstance(agent, dict):
                    self.run.log_param(agent)
                else:
                    add_hparams(agent, self.run)
            except Exception:
                # be permissive; logging hparams should not break the run
                pass

    def log_hparams(self, agent_or_hparams):
        """Log hyperparameters from an agent or a precomputed dict.

        Accepts either an object (will use `add_hparams`) or a plain dict.
        """
        if agent_or_hparams is None:
            return
        if isinstance(agent_or_hparams, dict):
            self.run.log_param(agent_or_hparams)
        else:
            add_hparams(agent_or_hparams, self.run)

    def __enter__(self):
        # Return a context object that supports both tuple-unpacking and
        # attribute access by delegating to TensorboardRun.
        return TensorboardContext(self.run, self.local_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        try:
            self.run.writer.close()
        except Exception:
            pass
