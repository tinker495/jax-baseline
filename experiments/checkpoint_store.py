import os
import pickle

import jax
import numpy as np


class FileCheckpointStore:
    def save(self, path, state):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "arrays.npy"), "wb") as handle:
            for leaf in jax.tree_util.tree_leaves(state):
                np.save(handle, leaf, allow_pickle=False)

        structure = jax.tree_util.tree_map(lambda _leaf: 0, state)
        with open(os.path.join(path, "tree.pkl"), "wb") as handle:
            pickle.dump(structure, handle)

    def restore(self, path):
        with open(os.path.join(path, "tree.pkl"), "rb") as handle:
            structure = pickle.load(handle)

        leaves, treedef = jax.tree.flatten(structure)
        with open(os.path.join(path, "arrays.npy"), "rb") as handle:
            values = [np.load(handle) for _ in leaves]
        return jax.tree.unflatten(treedef, values)
