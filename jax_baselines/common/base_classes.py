import glob
import os
import pickle

import jax
import numpy as np
import optax
from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run.

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.writer is None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            self.save_path = os.path.join(
                self.tensorboard_log_path,
                "{}_{}".format(self.tb_log_name, latest_run_id),
            )
            self.writer = SummaryWriter(self.save_path)
        return self.writer, self.save_path

    def _get_latest_run_id(self):
        """returns the latest run number for the given log name and log path, by finding the greatest number in the
        directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.tensorboard_log_path, self.tb_log_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if (
                self.tb_log_name == "_".join(file_name.split("_")[:-1])
                and ext.isdigit()
                and int(ext) > max_run_id
            ):
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.flush()


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
    elif optim_str == "lion":
        optim = optax.lion(lr, weight_decay=1e-5)

    if grad_max is not None:
        optim = optax.chain(optax.clip_by_global_norm(grad_max), optim)

    return optim
