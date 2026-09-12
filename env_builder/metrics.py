"""Adapter-owned scalar aggregation and Gymnasium episode diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.env_protocols import EnvironmentMetadata, EvaluationContextEnv
from jax_baselines.core.runtime_adapters import MetricLogger


def metric_leaves(values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten diagnostic groups while rejecting ambiguous metric names."""
    leaves: dict[str, Any] = {}

    def flatten(items: Mapping[str, Any], prefix: str = "") -> None:
        for name, value in items.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Environment metric names must be non-empty strings")
            key = f"{prefix}/{name}" if prefix else name
            if isinstance(value, Mapping):
                flatten(value, key)
            else:
                if key in leaves:
                    raise ValueError(f"Duplicate environment metric: {key}")
                leaves[key] = value

    flatten(values)
    return leaves


@jax.jit
def _pack_jax_metrics(values: tuple[jax.Array, ...]) -> jax.Array:
    return jnp.stack([value.mean() for value in values])


class EnvMetrics:
    """Keep bounded sums on the producing device; transfer only when logging."""

    def __init__(self, array_converter: Callable[[Any], np.ndarray] = np.asarray) -> None:
        self._array_converter = array_converter
        self._sums: dict[tuple[str, ...], Any] = {}
        self._weights: dict[tuple[str, ...], int] = {}

    def add(self, values: Mapping[str, Any], weight: int = 1) -> None:
        leaves = metric_leaves(values)
        if not leaves:
            return
        array_module = jnp if any(isinstance(v, jax.Array) for v in leaves.values()) else np
        arrays = []
        for key, value in leaves.items():
            array = array_module.asarray(value)
            if array.dtype.kind not in "biuf" or not array.size:
                raise ValueError(f"Environment metric {key!r} must contain real numeric values")
            arrays.append(array)
        self.add_batch(
            tuple(leaves),
            (
                _pack_jax_metrics(tuple(arrays))
                if array_module is jnp
                else np.stack([array.mean() for array in arrays])
            ),
            weight,
        )

    def add_batch(self, names: tuple[str, ...], values: Any, weight: int = 1) -> None:
        """Accept a packed native metric vector without per-term device operations."""
        if weight < 1 or values.shape != (len(names),):
            raise ValueError("Expected a numeric metric vector and a positive weight")
        if isinstance(values, (np.ndarray, jax.Array)) and values.dtype.kind not in "biuf":
            raise ValueError("Environment metrics must be real numeric values")
        if not names:
            return
        if names in self._sums:
            self._sums[names] = self._sums[names] + values * weight
            self._weights[names] += weight
        else:
            self._sums[names] = values * weight
            self._weights[names] = weight

    def log(self, logger: MetricLogger, steps: int | None, namespace: str) -> None:
        totals: dict[str, float] = {}
        weights: dict[str, int] = {}
        for names, packed in self._sums.items():
            values = self._array_converter(packed)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite environment metrics in {names}")
            for name, value in zip(names, values, strict=True):
                if name not in totals:
                    totals[name] = 0.0
                    weights[name] = 0
                totals[name] += float(value)
                weights[name] += self._weights[names]
        for name, total in totals.items():
            logger.log_metric(f"{namespace}/{name}", total / weights[name], steps)
        self._sums.clear()
        self._weights.clear()


class GymEnvMetrics:
    """Unclipped game returns and explicit ``info['log']`` diagnostics.

    Only action-applied transitions count. Life loss preserves game returns;
    true termination or truncation completes them. Evaluation's worker mask
    excludes completed quotas before accumulation.
    """

    def __init__(self, workers: int, reward_key: str = "original_reward") -> None:
        self._workers = workers
        self._reward_key = reward_key
        self._returns = np.zeros(workers, dtype=np.float64)
        self._present = np.zeros(workers, dtype=bool)
        self._autoreset = np.zeros(workers, dtype=bool)
        self._pending: tuple[Any, Any, Any, Any] | None = None
        self._metrics = EnvMetrics()

    def capture(self, infos: Any, done: Any, real_reset: Any, autoreset: Any) -> None:
        self._pending = infos, done, real_reset, autoreset

    def reset(self) -> None:
        self._returns.fill(0)
        self._present.fill(False)
        self._autoreset.fill(False)
        self._pending = None

    def _vector(self, value: Any, dtype: Any) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.shape == ():
            return np.full(self._workers, array.item(), dtype=dtype)
        if array.shape != (self._workers,):
            raise ValueError(
                f"Expected environment metric shape {(self._workers,)}, got {array.shape}"
            )
        return array

    def _record_info(
        self, info: Mapping[str, Any], active: np.ndarray, *, vectorized: bool = True
    ) -> None:
        if self._reward_key in info:
            present = active.copy()
            if f"_{self._reward_key}" in info:
                present &= self._vector(info[f"_{self._reward_key}"], bool)
            self._returns[present] += self._vector(info[self._reward_key], np.float64)[present]
            self._present |= present
        if "log" not in info:
            return
        if not isinstance(info["log"], Mapping):
            raise TypeError("Environment info['log'] must be a mapping")
        if "_log" in info:
            active = active & self._vector(info["_log"], bool)

        def record(items: Mapping[str, Any], selected: np.ndarray, prefix: str = "") -> None:
            for name, value in items.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Environment metric names must be non-empty strings")
                if name.startswith("_"):
                    continue
                valid = selected
                if f"_{name}" in items:
                    valid = valid & self._vector(items[f"_{name}"], bool)
                if not valid.any():
                    continue
                key = f"{prefix}/{name}" if prefix else name
                if isinstance(value, Mapping):
                    record(value, valid, key)
                    continue
                array = np.asarray(value)
                if vectorized and array.ndim and array.shape[0] == self._workers:
                    array = array[valid]
                self._metrics.add({key: array}, weight=int(valid.sum()))

        record(info["log"], active)

    def log(
        self,
        logger: MetricLogger,
        steps: int | None,
        *,
        namespace: str = "rollout",
        active: Any = None,
        flush: bool = True,
    ) -> None:
        if self._pending is not None:
            infos, done, real_reset, autoreset = self._pending
            self._pending = None
            selected = ~self._autoreset
            if active is not None:
                selected &= self._vector(active, bool)
            if isinstance(infos, Mapping):
                self._record_info(infos, selected)
            elif isinstance(infos, (list, tuple)):
                if len(infos) != self._workers:
                    raise ValueError("Expected one environment info mapping per worker")
                for worker, info in enumerate(infos):
                    if not isinstance(info, Mapping):
                        raise TypeError("Environment info must be a mapping")
                    mask = np.zeros(self._workers, dtype=bool)
                    mask[worker] = selected[worker]
                    self._record_info(info, mask, vectorized=False)
            else:
                raise TypeError("Environment info must be a mapping or per-worker mappings")
            finished = self._vector(done, bool) & self._vector(real_reset, bool) & selected
            emit = finished & self._present
            if emit.any():
                self._metrics.add({"original_reward": self._returns[emit]}, weight=int(emit.sum()))
            self._returns[finished] = 0
            self._present[finished] = False
            self._autoreset = (
                self._vector(done, bool) & self._vector(autoreset, bool) & ~self._autoreset
            )
        if flush:
            self._metrics.log(logger, steps, namespace)


class GymLoggingWrapper(gym.Wrapper):
    """Expose diagnostics on the outer single-environment protocol surface."""

    def __init__(
        self, env: gym.Env, *, env_id: str, seed: int | None, is_atari: bool = False
    ) -> None:
        super().__init__(env)
        self._is_atari = is_atari
        self._metrics = GymEnvMetrics(1)
        self.runtime: EnvironmentMetadata = {
            "backend": "gymnasium",
            "backend_env_id": env_id,
            "seed": seed,
            "seed_rule": "reset seed; action and observation spaces seeded with the same seed",
            "reward_clipping": "sign" if is_atari else "none",
            "episodic_life": is_atari,
        }

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._metrics.capture(
            info,
            terminated or truncated,
            bool(info["real_episode_end"])
            if "real_episode_end" in info
            else bool(terminated or truncated),
            False,
        )
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._metrics = GymEnvMetrics(1)
        return self.env.reset(seed=seed, options=options)

    def reset_for_evaluation(self):
        self._metrics = GymEnvMetrics(1)
        if self._is_atari:
            self.env.set_wrapper_attr("was_real_done", True, force=False)
        return self.reset()

    @contextmanager
    def evaluation_context(self) -> Iterator[None]:
        metrics = self._metrics
        with (
            self.env.evaluation_context()
            if isinstance(self.env, EvaluationContextEnv)
            else nullcontext()
        ):
            self._metrics = GymEnvMetrics(1)
            try:
                yield
            finally:
                self._metrics = metrics

    def log_metrics(
        self,
        logger: MetricLogger,
        steps: int | None,
        *,
        namespace: str = "rollout",
        active: Any = None,
        flush: bool = True,
    ) -> None:
        self._metrics.log(logger, steps, namespace=namespace, active=active, flush=flush)
