"""Core-owned replay factory protocols.

Concrete replay storage lives in adapter packages such as :mod:`replay_memory`.
Algorithm-family code depends on these minimal callable seams instead of naming
cpprb-backed implementations directly.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import jax

from jax_baselines.core.env_protocols import Observation


def select_replay_device(observations: Observation, *, required: bool) -> jax.Device | None:
    """Follow a single GPU observation device, or require an available GPU."""
    devices: set[jax.Device] = set()
    for value in observations.values():
        if not isinstance(value, jax.Array):
            devices.clear()
            break
        devices.update(value.devices())
    if len(devices) == 1:
        device = devices.pop()
        if device.platform == "gpu":
            return device
    if not required:
        return None
    try:
        return jax.devices("gpu")[0]
    except (RuntimeError, IndexError) as error:
        raise ValueError("memory_backend='gpu' requires a JAX GPU device") from error


@dataclass(frozen=True)
class PriorityNeed:
    alpha: float
    eps: float


class ReplayWriter(Protocol):
    def add(
        self,
        obs_t,
        action,
        reward,
        nxtobs_t,
        terminated,
        truncated=False,
        store_mask=None,
    ) -> None:
        ...


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
    memory_backend: Literal["cpu", "gpu"] = "cpu"
    device: jax.Device | None = None
    seed: int = 0


@dataclass(frozen=True, kw_only=True)
class SelfPredictionReplayNeed(LocalReplayNeed):
    prediction_depth: int


@dataclass(frozen=True, kw_only=True)
class SharedPrioritizedReplayNeed:
    buffer_size: int
    observation_space: Any
    action_shape_or_n: Any
    n_step: int
    gamma: float
    manager: Any
    priority: PriorityNeed
    compress_observations: bool = False


class ReplayBufferFactory(Protocol):
    def __call__(self, need: LocalReplayNeed) -> Any:
        ...


class WorkerReplayBufferFactory(Protocol):
    def __call__(self, local_size: int, *, env_dict: dict, n_s: dict | None = None) -> Any:
        ...


@dataclass(frozen=True)
class ApeXReplayTopology:
    shared_buffer: Any
    worker_factory: WorkerReplayBufferFactory


class ApeXReplayFactory(Protocol):
    def __call__(self, need: SharedPrioritizedReplayNeed) -> ApeXReplayTopology:
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
