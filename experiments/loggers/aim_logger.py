"""Aim logging backend (``--logger aim``).

Satisfies :class:`jax_baselines.core.runtime_adapters.LoggerRun`. ``aim`` is an
optional dependency (it ships in the ``dev`` group, so ``uv sync`` installs it);
it is imported lazily by :func:`make_aim_logger_factory` only when
``--logger aim`` is selected, so the core never needs it. Selecting it
uninstalled raises a clear, actionable error.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import numpy as np

from experiments.run_metadata import tracking_experiment_name, write_run_metadata
from experiments.runtime_adapters import create_run_directory
from jax_baselines.core.hparams import get_hyper_params


def _import_aim():
    try:
        import aim
    except ImportError as exc:
        raise SystemExit(
            "--logger aim requires the optional 'aim' dependency, which is not installed.\n"
            "Install it with:  uv sync   (it ships in the dev group; or:  uv pip install aim)"
        ) from exc
    return aim


class AimRun:
    """A :class:`LoggerRun` backed by one live Aim ``Run``."""

    def __init__(
        self, aim_module, run, local_dir: str, extra_hparams: dict[str, str] | None = None
    ):
        self._aim = aim_module
        self._run = run
        self._dir = local_dir
        self._extra_hparams = {} if extra_hparams is None else dict(extra_hparams)

    def log_param(self, hparam_dict):
        self._run["hparams"] = {**hparam_dict, **self._extra_hparams}

    def log_metric(self, key, value, step=None):
        # Aim requires a Python number; the core emits JAX/NumPy scalars.
        self._run.track(float(value), name=key, step=step)

    def log_histogram(self, key, value, step=None):
        self._run.track(self._aim.Distribution(np.asarray(value)), name=key, step=step)

    def declare_multiline_layout(self, eps):
        # Aim groups metrics by name in its own UI; it has no TensorBoard custom-
        # scalars layout concept, so the per-epsilon multiline overlay is a no-op.
        return None

    def get_local_path(self, path):
        return os.path.join(self._dir, path)


class AimLogger:
    """LoggerFactory + context manager mapping a training run to one Aim Run.

    ``__enter__`` is idempotent: the distributed Ray logger actor re-enters per
    call, and every entry returns the same live Run, so the centralized actor
    maps to a single Aim Run. The Run is closed on ``close`` / ``__del__``.
    """

    def __init__(
        self,
        aim_module,
        run_name: str,
        experiment_name: str,
        local_dir: str,
        agent: Any | None,
        *,
        repo: str | None = None,
        extra_hparams: dict[str, str] | None = None,
        run_metadata: dict[str, object] | None = None,
    ):
        self._aim = aim_module
        self._run_name = run_name
        self._experiment_name = tracking_experiment_name(experiment_name, run_metadata)
        self._local_dir = create_run_directory(local_dir, experiment_name, run_name)
        write_run_metadata(self._local_dir, run_metadata)
        self._repo = repo
        self._agent = agent
        self._extra_hparams = extra_hparams
        self._run = None
        self._logger_run: AimRun | None = None

    def _ensure_started(self) -> AimRun:
        if self._logger_run is not None:
            return self._logger_run
        self._run = self._aim.Run(repo=self._repo, experiment=self._experiment_name)
        self._run.name = self._run_name
        self._logger_run = AimRun(self._aim, self._run, self._local_dir, self._extra_hparams)
        if self._agent is not None:
            self.log_hparams(self._agent)
        return self._logger_run

    def log_hparams(self, agent_or_hparams):
        if agent_or_hparams is None:
            return
        run = self._ensure_started()
        hparams = (
            agent_or_hparams
            if isinstance(agent_or_hparams, dict)
            else get_hyper_params(agent_or_hparams)
        )
        run.log_param(hparams)

    def __enter__(self) -> AimRun:
        return self._ensure_started()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Keep the Run open across re-entries; it is closed on close()/__del__.
        return False

    def close(self):
        if self._run is not None:
            self._run.close()
            self._run = None
            self._logger_run = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_aim_logger_factory(
    args,
    *,
    extra_hparams: dict[str, str] | None = None,
    run_metadata: dict[str, object] | None = None,
):
    """Return a ``LoggerFactory`` for Aim, capturing the CLI backend config.

    The ``aim`` import happens here (only when ``--logger aim`` is selected), so
    an uninstalled extra fails fast with a clear message before training or
    ``ray.init()`` starts.
    """
    return partial(
        AimLogger,
        _import_aim(),
        repo=args.aim_repo,
        extra_hparams=extra_hparams,
        run_metadata=run_metadata,
    )
