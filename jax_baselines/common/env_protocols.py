"""Core-facing environment compatibility contracts.

The concrete Gymnasium/EnvPool adapter lives in the repo-local ``env_builder``
package.  Algorithm-core utilities depend on the small contracts in this module
instead of concrete backend packages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypedDict, runtime_checkable


class EnvInfo(TypedDict):
    """Vectorized-environment metadata shared across adapter and core layers."""

    observation_space: Any
    action_space: Any
    env_type: str
    env_id: str


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


class Env(ABC):
    """Async-style env surface used by vectorized adapters."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def current_obs(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> None:
        pass

    @abstractmethod
    def get_result(self) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class VectorizedEnv(Env):
    """Core-facing marker for vectorized environment adapters."""

    env_info: EnvInfo | None = None
