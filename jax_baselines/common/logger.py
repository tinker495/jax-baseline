import glob
import os
import mlflow


def _get_latest_run_id(local_dir, run_name):
    """returns the latest run number for the given log name and log path, by finding the greatest number in the
    directories.

    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob("{}/{}_[0-9]*".format(local_dir, run_name)):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if (
            run_name == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id

class MLflowRun:
    def __init__(self, run_name: str, local_dir: str):
        self.run = mlflow.start_run(run_name=run_name)
        self.local_dir = local_dir

    def log_tag(self, tags: dict):
        mlflow.set_tags(tags)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step)

    def log_figure(self, figure, name):
        mlflow.log_figure(figure, name)

    def get_local_path(self, path):
        return os.path.join(self.local_dir, path)

    def log_artifact(self, path):
        mlflow.log_artifact(self.get_local_path(path))

class MLflowLogger:
    def __init__(self, 
                 run_name: str,
                 experiment_name: str,
                 local_dir: str,
                 tags: dict = {},
                 tracking_uri: str="http://0.0.0.0:6006"):
        """
        Create an MLflow logger for a code segment, and saves it to the MLflow server as its own run.

        :param mlflow_tracking_uri: (str) the tracking URI for MLflow
        :param mlflow_experiment_name: (str) the name of the experiment for MLflow logging
        """
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.local_dir = os.path.join(
                local_dir,
                "{}_{}".format(run_name, _get_latest_run_id(local_dir, run_name)),
            )
        tags["local_dir"] = self.local_dir
        self.tags = tags
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = MLflowRun(self.run_name, self.local_dir)
        mlflow.set_tags(self.tags)

    def __enter__(self) -> MLflowRun:
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        if self.run is not None:
            mlflow.end_run()
