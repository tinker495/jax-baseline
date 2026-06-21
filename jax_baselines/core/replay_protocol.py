"""Core-owned replay factory protocols.

Concrete replay storage lives in adapter packages such as :mod:`replay_memory`.
Algorithm-family code depends on these minimal callable seams instead of naming
cpprb-backed implementations directly.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PriorityNeed:
    alpha: float
    eps: float


@dataclass(frozen=True, kw_only=True)
class LocalReplayNeed:
    buffer_size: int
    observation_space: Any
    action_shape_or_n: Any
    worker_size: int = 1
    n_step: int = 1
    gamma: float = 0.99
    priority: PriorityNeed | None = None
    compress_observations: bool = False
    n_frames: int = 4


@dataclass(frozen=True, kw_only=True)
class SelfPredictionReplayNeed(LocalReplayNeed):
    prediction_depth: int


class ReplayBufferFactory(Protocol):
    def __call__(self, need: LocalReplayNeed) -> Any:
        ...


class MultiPrioritizedReplayBufferFactory(Protocol):
    def __call__(
        self,
        *,
        buffer_size: int,
        observation_space: Any,
        alpha: float,
        action_shape_or_n: Any,
        n_step: int,
        gamma: float,
        manager: Any,
        compress_memory: bool = False,
        eps: float | None = None,
    ) -> Any:
        ...


class WorkerReplayBufferFactory(Protocol):
    def __call__(self, local_size: int, *, env_dict: dict, n_s: dict | None = None) -> Any:
        ...


def require_replay_factory(factory: Any, role: str) -> Any:
    """Return ``factory`` or fail fast when composition omitted the adapter."""
    if factory is None:
        raise ValueError(
            f"{role} is required. Supply a concrete replay factory from the "
            "experiments/replay_memory adapter composition layer."
        )
    return factory


def make_worker_local_replay_buffer(
    worker_replay_factory: WorkerReplayBufferFactory | None,
    local_size: int,
    env_dict: dict,
    n_s: dict | None,
) -> Any:
    factory = require_replay_factory(worker_replay_factory, "WorkerReplayBufferFactory")
    return factory(local_size, env_dict=env_dict, n_s=n_s)
