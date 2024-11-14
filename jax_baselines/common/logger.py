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

    def log_custom_scalars(self, layout):
        self.writer.add_custom_scalars(layout)


class TensorboardLogger:
    def __init__(self, run_name: str, experiment_name: str, local_dir: str, agent: any):
        """
        Create an MLflow logger for a code segment, and saves it to the MLflow server as its own run.

        :param mlflow_tracking_uri: (str) the tracking URI for MLflow
        :param mlflow_experiment_name: (str) the name of the experiment for MLflow logging
        """
        self.run_name = run_name
        self.local_dir = os.path.join(
            local_dir,
            experiment_name,
            f"{run_name}_{_get_latest_run_id(local_dir, experiment_name, run_name)+1:02d}",
        )
        self.run = TensorboardRun(self.local_dir)
        add_hparams(agent, self.run)

    def get_run(self):
        return self.run

    def __enter__(self) -> TensorboardRun:
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        self.run.writer.close()
