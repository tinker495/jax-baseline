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
    """Minimal live logger surface the core uses inside a run."""

    def log_param(self, hparam_dict: dict) -> None:
        ...

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        ...

    def get_local_path(self, path: str) -> str:
        ...


class HistogramLoggerRun(LoggerRun, Protocol):
    def log_histogram(self, key: str, value: Any, step: Optional[int] = None) -> None:
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


class NoOpWriter:
    """SummaryWriter-shaped no-op for core-only runs and distributed tests."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_custom_scalars(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_summary(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class NoOpLoggerRun:
    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        self.writer = NoOpWriter()

    def __iter__(self) -> Iterable[Any]:
        yield self.writer
        yield self.local_dir

    def log_param(self, hparam_dict: dict) -> None:
        return None

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        return None

    def log_histogram(self, key: str, value: Any, step: Optional[int] = None) -> None:
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
