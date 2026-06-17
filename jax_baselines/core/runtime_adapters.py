"""Core runtime adapter protocols and no-op defaults.

Direct core callers that do not inject experiments adapters get no TensorBoard,
no tqdm progress display, and no video recording. The repository's user-facing
experiment runners inject concrete implementations from ``experiments``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LoggerRun(Protocol):
    """The stable contract the Algorithm Core logs through inside a run.

    Every backend implements the whole surface. ``log_histogram`` and
    ``declare_multiline_layout`` degrade to no-ops on backends that do not
    support them, so call sites never feature-test the logger or reach for a
    backend-specific writer.
    """

    def log_param(self, hparam_dict: dict) -> None:
        ...

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        ...

    def log_histogram(self, key: str, value: Any, step: Optional[int] = None) -> None:
        ...

    def declare_multiline_layout(self, eps: Iterable[float]) -> None:
        """Declare the per-epsilon rollout multiline layout (TensorBoard custom
        scalars). Backends without a custom-layout concept no-op."""
        ...

    def get_local_path(self, path: str) -> str:
        ...


class LoggerFactory(Protocol):
    def __call__(
        self, run_name: str, experiment_name: str, local_dir: str, agent: Optional[Any]
    ) -> Any:
        ...


class ProgressBar(Protocol):
    def __iter__(self) -> Iterator[int]:
        ...

    def set_description(self, desc: str) -> None:
        ...


class ProgressFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> ProgressBar:
        ...


class EvaluationRecorder(Protocol):
    def __call__(self, env_builder, logger_run, actions_eval_fn, episode, conv_action=None):
        ...


class NoOpLoggerRun:
    """Protocol-safe ``LoggerRun`` used when no experiment adapter is injected."""

    def __init__(self, local_dir: str):
        self.local_dir = local_dir

    def log_param(self, hparam_dict: dict) -> None:
        return None

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        return None

    def log_histogram(self, key: str, value: Any, step: Optional[int] = None) -> None:
        return None

    def declare_multiline_layout(self, eps: Iterable[Any]) -> None:
        return None

    def get_local_path(self, path: str) -> str:
        return os.path.join(self.local_dir, path)


class NoOpLogger:
    """Protocol-safe logger used when no experiment adapter is injected."""

    def __init__(self, run_name: str, experiment_name: str, local_dir: str, agent: Optional[Any]):
        self.run = NoOpLoggerRun(local_dir)

    def log_hparams(self, agent_or_hparams: Any) -> None:
        return None

    def __enter__(self) -> NoOpLoggerRun:
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def close(self) -> None:
        return None


class NoOpProgress:
    def __init__(self, *args: Any, **kwargs: Any):
        self._range = range(*args)

    def __iter__(self) -> Iterator[int]:
        return iter(self._range)

    def set_description(self, desc: str) -> None:
        return None


def make_progress(*args: Any, **_kwargs: Any) -> NoOpProgress:
    """Core progress factory; experiments injects tqdm for user-facing runs."""

    return NoOpProgress(*args)
