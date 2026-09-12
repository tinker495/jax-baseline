"""Core-facing environment compatibility contracts.

The concrete Gymnasium/EnvPool adapter lives in the repo-local ``env_builder``
package.  Algorithm-core utilities depend on the small contracts in this module
instead of concrete backend packages.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.runtime_adapters import MetricLogger

# Keys retain their role from the environment boundary through replay and model input:
# unified_* is shared, actor_* is policy-only, and critic_* is value-only.
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
    autoreset_steps: NotRequired[bool]


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

    def get_info(self) -> EnvInfo:
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


@runtime_checkable
class VectorizedEvalEnv(VectorizedEnv, Protocol):
    """Vector environment that can start an independent evaluation measurement."""

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        ...


@runtime_checkable
class EvaluationContextEnv(Protocol):
    """Adapter-owned evaluation isolation, including restoration on failure.

    Shared train/eval environments must preserve their simulator, task, RNG,
    observation caches and any completed result waiting for collection.
    """

    def evaluation_context(self) -> AbstractContextManager[None]:
        ...


@runtime_checkable
class EnvironmentLogging(Protocol):
    """Environment-owned diagnostics, independent of algorithm observations/rewards.

    Call after each consumed transition, before reset or evaluation. ``active``
    selects evaluation workers; adapters exclude their own autoreset dummy rows.
    ``flush=False`` accumulates without logging or host synchronization. A flush
    without a new transition only emits pending aggregates. Adapters never retain
    the logger, and shared evaluation must isolate their diagnostic state too.
    """

    def log_metrics(
        self,
        logger: MetricLogger,
        steps: int | None,
        *,
        namespace: str = "rollout",
        active: Any = None,
        flush: bool = True,
    ) -> None:
        ...


def log_environment_metrics(
    env: Any,
    logger: MetricLogger | None,
    steps: int | None,
    *,
    namespace: str = "rollout",
    active: Any = None,
    flush: bool = True,
) -> None:
    """Dispatch optional diagnostics through the environment protocol."""
    if logger is not None and isinstance(env, EnvironmentLogging):
        env.log_metrics(logger, steps, namespace=namespace, active=active, flush=flush)


# Backward-compatible name exported by env_builder; no separate ABC needed.
Env = VectorizedEnv


def batch_observation(observation: Observation) -> Observation:
    """Add the leading model batch dimension to a single observation."""
    if not isinstance(observation, dict):
        raise TypeError("observation must be a dict")
    return {key: value[None, ...] for key, value in observation.items()}


def _done_mask(terminateds: Any, truncateds: Any) -> np.ndarray | jax.Array:
    if isinstance(terminateds, jax.Array):
        return jnp.logical_or(terminateds, truncateds)
    return np.logical_or(
        np.asarray(terminateds, dtype=bool),
        np.asarray(truncateds, dtype=bool),
    )


def vector_autoreset_mask(
    env: Any, terminateds: Any, truncateds: Any, infos: Any
) -> np.ndarray | jax.Array:
    autoreset_mask = getattr(env, "autoreset_mask", None)
    if callable(autoreset_mask):
        mask = autoreset_mask(terminateds, truncateds, infos)
        return mask.astype(bool) if isinstance(mask, jax.Array) else np.asarray(mask, dtype=bool)
    return _done_mask(terminateds, truncateds)


def reset_for_evaluation(env: Any) -> Any:
    """Reset an adapter for an independent evaluation run."""
    reset = getattr(env, "reset_for_evaluation", None)
    return reset() if callable(reset) else env.reset()
