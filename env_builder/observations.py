"""Normalize backend observations to the core's flat dict contract."""

from collections.abc import Mapping

import numpy as np


def _to_numpy(value):
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected observation must be one numeric array") from exc


def _numeric_array(value):
    array = _to_numpy(value)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Selected observation must be one numeric array")
    return array


def _children(value):
    children = getattr(value, "spaces", value)
    if isinstance(children, Mapping):
        return children
    if isinstance(children, tuple):
        return {str(index): child for index, child in enumerate(children)}
    return None


def _select_path(value, path, kind="Observation"):
    for part in path.split("."):
        children = _children(value)
        if children is None:
            raise ValueError(f"{kind} path {path!r} crosses a non-mapping at {part!r}")
        if part not in children:
            keys = ", ".join(children) or "<none>"
            raise KeyError(
                f"{kind} key {part!r} not found while selecting {path!r}; available keys: {keys}"
            )
        value = children[part]
    return value


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


def normalize_observation(observation, observation_key=None):
    """Flatten raw shared observations and mark every leaf with ``unified_``."""
    if observation_key:
        observation = _select_path(observation, observation_key)
    normalized = _flatten(observation, _numeric_array, prefix=observation_key or "")
    if not normalized:
        raise ValueError("Observation must contain at least one array leaf")
    return {f"unified_{key}": value for key, value in sorted(normalized.items())}


def normalize_observation_space(space, observation_key=None):
    """Return flattened observation shapes with the same keys as observations."""
    return {
        key: list(getattr(leaf, "shape", leaf))
        for key, leaf in flatten_observation_space(space, observation_key).items()
    }


def flatten_observation_space(space, observation_key=None):
    """Return flattened leaf spaces keyed like :func:`normalize_observation`."""
    if observation_key:
        space = _select_path(space, observation_key, "Observation-space")
    normalized = dict(
        sorted(
            _flatten(
                space,
                kind="Observation-space",
                prefix=observation_key or "",
            ).items()
        )
    )
    if not normalized:
        raise ValueError("Observation space must contain at least one leaf")
    return {f"unified_{key}": value for key, value in normalized.items()}
