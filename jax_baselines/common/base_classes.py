import glob
import os
import pickle
import mlflow

import jax
import numpy as np
import optax


class MLflowLogger:
    def __init__(self, tracking_uri="http://0.0.0.0:5000",
                 registry_uri="http://0.0.0.0:5000",
                 experiment_name="default",
                 tags=None,
                 tracking_token=None,
                 save_artifact=True,
                 new_mlflow_run=True):
        """
        Create an MLflow logger for a code segment, and saves it to the MLflow server as its own run.

        :param mlflow_tracking_uri: (str) the tracking URI for MLflow
        :param mlflow_experiment_name: (str) the name of the experiment for MLflow logging
        :param new_mlflow_run: (bool) whether or not to create a new run for MLflow
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        self.tags = tags
        self.tracking_token = tracking_token
        self.should_save_artifact = save_artifact
        self.new_mlflow_run = new_mlflow_run
        self.run = None

    def __enter__(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.set_tags(self.tags)
        mlflow.set_tracking_token(self.tracking_token)
        if self.new_mlflow_run:
            self.run = mlflow.start_run()
        else:
            self.run = mlflow.active_run()
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run is not None:
            mlflow.end_run()

def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def select_optimizer(optim_str, lr, eps=1e-2 / 256.0, grad_max=None):
    optim = None
    if optim_str == "adam":
        optim = optax.adam(lr, b1=0.9, b2=0.999, eps=eps)
    elif optim_str == "adamw":
        optim = optax.adamw(lr, b1=0.9, b2=0.999, eps=eps, weight_decay=1e-4)
    elif optim_str == "rmsprop":
        optim = optax.rmsprop(lr, eps=eps)
    elif optim_str == "sgd":
        optim = optax.sgd(lr)
    elif optim_str == "adabelief":
        optim = optax.adabelief(lr, eps=eps)
    elif optim_str == "lion":
        optim = optax.lion(lr, weight_decay=1e-5)

    if grad_max is not None:
        optim = optax.chain(optax.clip_by_global_norm(grad_max), optim)

    return optim
