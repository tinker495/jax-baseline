"""Normalize backend observations to the core's flat dict contract."""

from collections.abc import Mapping

import numpy as np


def _to_numpy(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Observation must be one numeric array") from exc
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Observation must be one numeric array")
    return array


def _children(value):
    children = getattr(value, "spaces", value)
    if isinstance(children, Mapping):
        return children
    if isinstance(children, tuple):
        return {str(index): child for index, child in enumerate(children)}
    return None


def _flatten(value, leaf=None, kind="Observation", prefix=""):
    children = _children(value)
    if children is None:
        return {prefix or "obs": leaf(value) if leaf else value}

    leaves = {}
    for key, child in children.items():
        if not isinstance(key, str):
            raise TypeError(f"{kind} keys must be strings")
        path = f"{prefix}.{key}" if prefix else key
        for child_path, child_leaf in _flatten(child, leaf, kind, path).items():
            if child_path in leaves:
                raise ValueError(f"Duplicate {kind.lower()} path: {child_path!r}")
            leaves[child_path] = child_leaf
    return leaves


def normalize_observation(observation):
    """Return a deterministic ``dict[str, ndarray]`` for every backend."""
    normalized = _flatten(observation, _to_numpy)
    if not normalized:
        raise ValueError("Observation must contain at least one array leaf")
    return dict(sorted(normalized.items()))


def normalize_observation_space(space):
    """Return flattened observation shapes with the same keys as observations."""
    return {
        key: list(getattr(leaf, "shape", leaf))
        for key, leaf in flatten_observation_space(space).items()
    }


def flatten_observation_space(space):
    """Return flattened leaf spaces keyed like :func:`normalize_observation`."""
    normalized = dict(sorted(_flatten(space, kind="Observation-space").items()))
    if not normalized:
        raise ValueError("Observation space must contain at least one leaf")
    return normalized
