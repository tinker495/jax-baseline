"""Core-facing environment compatibility contracts.

The concrete Gymnasium/EnvPool adapter lives in the repo-local ``env_builder``
package.  Algorithm-core utilities depend on the small contracts in this module
instead of concrete backend packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

import numpy as np

Observation: TypeAlias = dict[str, Any]
ObservationSpace: TypeAlias = dict[str, list[int]]


class EnvInfo(TypedDict):
    """Environment metadata shared across adapter and core layers."""

    observation_space: ObservationSpace
    action_size: list[int]
    action_type: Literal["discrete", "continuous"]
    env_type: str
    env_id: str
    worker_num: int
    core_env_type: str


@dataclass(frozen=True)
class PreparedEnvSpec:
    """Adapter-prepared train/eval environments plus typed metadata."""

    env: Any
    eval_env: Any
    env_info: EnvInfo


@dataclass(frozen=True)
class PreparedWorkerEnvSpec:
    """Adapter-prepared single worker environment plus typed metadata."""

    env: Any
    env_info: EnvInfo


@runtime_checkable
class SingleEnv(Protocol):
    """Single-environment surface consumed by core metadata helpers."""

    observation_space: Any
    action_space: Any

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Observation, dict[str, Any]]:
        ...

    def step(self, action: Any) -> tuple[Observation, Any, Any, Any, dict[str, Any]]:
        ...

    def close(self) -> Any:
        ...


@runtime_checkable
class VectorizedEnv(Protocol):
    """Async-style vectorized environment surface consumed by the core.

    ``get_result()[0]`` is the successor for the completed transition, while
    ``current_obs()`` is the observation for the next action. They may differ
    when an adapter performs same-step autoreset.
    """

    env_info: EnvInfo | None = None

    def get_info(self) -> dict[str, Any]:
        ...

    def current_obs(self) -> Observation:
        """Return the observation corresponding to the next action."""
        ...

    def step(self, action: Any) -> None:
        ...

    def get_result(self) -> tuple[Observation, Any, Any, Any, Any]:
        """Return the completed transition, including its observation successor."""
        ...

    def close(self) -> None:
        ...


# Backward-compatible name exported by env_builder; no separate ABC needed.
Env = VectorizedEnv


def batch_observation(observation: Observation) -> Observation:
    """Add the leading model batch dimension to a single observation."""
    if not isinstance(observation, dict):
        raise TypeError("observation must be a dict")
    return {key: np.expand_dims(value, axis=0) for key, value in observation.items()}


def _done_mask(terminateds: Any, truncateds: Any) -> np.ndarray:
    return np.logical_or(
        np.asarray(terminateds, dtype=bool),
        np.asarray(truncateds, dtype=bool),
    )


def vector_real_reset_mask(env: Any, terminateds: Any, truncateds: Any, infos: Any) -> np.ndarray:
    real_reset_mask = getattr(env, "real_reset_mask", None)
    if callable(real_reset_mask):
        return np.asarray(real_reset_mask(terminateds, truncateds, infos), dtype=bool)
    return _done_mask(terminateds, truncateds)


def vector_autoreset_mask(env: Any, terminateds: Any, truncateds: Any, infos: Any) -> np.ndarray:
    autoreset_mask = getattr(env, "autoreset_mask", None)
    if callable(autoreset_mask):
        return np.asarray(autoreset_mask(terminateds, truncateds, infos), dtype=bool)
    return _done_mask(terminateds, truncateds)


def single_real_episode_end(terminated: Any, truncated: Any, info: Any) -> bool:
    """Return the adapter-normalized real episode boundary for one environment."""
    if isinstance(info, dict) and "real_episode_end" in info:
        return bool(info["real_episode_end"])
    return bool(terminated or truncated)


def reset_for_evaluation(env: Any) -> Any:
    """Reset an adapter for an independent evaluation run."""
    reset = getattr(env, "reset_for_evaluation", None)
    return reset() if callable(reset) else env.reset()
