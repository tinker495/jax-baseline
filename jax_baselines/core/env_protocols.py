"""Core-facing environment compatibility contracts.

The concrete Gymnasium/EnvPool adapter lives in the repo-local ``env_builder``
package.  Algorithm-core utilities depend on the small contracts in this module
instead of concrete backend packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


class EnvInfo(TypedDict):
    """Environment metadata shared across adapter and core layers."""

    observation_space: list[list[int]]
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

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...

    def close(self) -> Any:
        ...


@runtime_checkable
class VectorizedEnv(Protocol):
    """Async-style vectorized environment surface consumed by the core."""

    env_info: EnvInfo | None = None

    def get_info(self) -> dict[str, Any]:
        ...

    def current_obs(self) -> Any:
        ...

    def step(self, action: Any) -> None:
        ...

    def get_result(self) -> Any:
        ...

    def close(self) -> None:
        ...


# Backward-compatible name exported by env_builder; no separate ABC needed.
Env = VectorizedEnv
