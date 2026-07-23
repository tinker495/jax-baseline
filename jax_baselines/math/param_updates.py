from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

PyTree = Any


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree):
    treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree, mul=1.2):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t) * mul,
        target,
        keys_tree,
    )


def scaled_by_reset(
    tensors: PyTree,
    optimizer_state: optax.GradientTransformationExtraArgs,
    optimizer: optax.GradientTransformation,
    key: jax.random.PRNGKey,
    steps: int,
    update_period: int,
    tau: float,
):
    update = steps % update_period == 0

    def _soft_reset(old_tensors, old_optimizer_state, key):
        new_tensors = tree_random_normal_like(key, tensors)
        soft_reseted = jax.tree_util.tree_map(
            lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
        )
        return soft_reseted, optimizer.init(soft_reseted)

    tensors, optimizer_state = jax.lax.cond(
        update, _soft_reset, lambda p, os, k: (p, os), tensors, optimizer_state, key
    )
    return tensors, optimizer_state


def scaled_by_reset_with_filter(
    tensors: PyTree,
    optimizer_state: optax.GradientTransformationExtraArgs,
    optimizer: optax.GradientTransformation,
    key: jax.random.PRNGKey,
    steps: int,
    update_period: int,
    taus: PyTree,
):
    update = steps % update_period == 0

    def _soft_reset(old_tensors, old_optimizer_state, key):
        new_tensors = tree_random_normal_like(key, tensors)
        soft_reseted = jax.tree_util.tree_map(
            lambda new, old, tau: tau * new + (1.0 - tau) * old,
            new_tensors,
            old_tensors,
            taus,
        )
        return soft_reseted, optimizer.init(soft_reseted)

    tensors, optimizer_state = jax.lax.cond(
        update, _soft_reset, lambda p, os, k: (p, os), tensors, optimizer_state, key
    )
    return tensors, optimizer_state


def filter_like_tree(tensors: PyTree, name_filter: str, filter_fn: Callable):
    # name_filter: "qnet"
    # filter_fn: lambda x,filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2
    # make a new tree with the same structure as the input tree, but with the values filtered by the filter_fn
    # for making tau = 1.0 for qnet and tau = 0.2 for the rest
    # this making hard_reset for qnet and scaled_by_reset for the rest
    def sigma_filter(x, sigma):
        return jnp.zeros_like(x) if sigma else x

    # noisynet's sigma is 0.0 for did not reset noisynet noise
    tensors = jax.tree_util.tree_map(lambda x: x, tensors)

    def _filter_like_tree(tensors, filtered: bool):
        for name, value in tensors.items():
            if isinstance(value, dict):
                tensors[name] = _filter_like_tree(value, filtered or name_filter in name)
            else:
                tensors[name] = sigma_filter(filter_fn(value, filtered), "sigma" in name)
        return tensors

    return _filter_like_tree(tensors, False)


def hard_update(new_tensors: PyTree, old_tensors: PyTree, steps: int, update_period: int):
    update = steps % update_period == 0
    return jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(update, new, old), new_tensors, old_tensors
    )


def soft_update(new_tensors: PyTree, old_tensors: PyTree, tau: float):
    return jax.tree_util.tree_map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
    )


def project_dense_kernels(tensors: PyTree, epsilon: float = 1e-8):
    """Project every Dense kernel column in a Flax variable tree to unit norm."""

    def project(path, value):
        keys = [getattr(entry, "key", None) for entry in path]
        if keys[-1] == "kernel" and any(
            isinstance(key, str) and key.startswith("Dense_") for key in keys[:-1]
        ):
            return value / jnp.maximum(jnp.linalg.norm(value, axis=0, keepdims=True), epsilon)
        return value

    return jax.tree_util.tree_map_with_path(project, tensors)
