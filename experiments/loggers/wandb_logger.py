"""Weights & Biases logging backend (``--logger wandb``).

Satisfies :class:`jax_baselines.core.runtime_adapters.LoggerRun`. ``wandb`` is an
optional dependency (it ships in the ``dev`` group, so ``uv sync`` installs it);
it is imported lazily by :func:`make_wandb_logger_factory` only when
``--logger wandb`` is selected, so the core never needs it. Selecting it
uninstalled raises a clear, actionable error.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from jax_baselines.core.hparams import get_hyper_params


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "--logger wandb requires the optional 'wandb' dependency, which is not installed.\n"
            "Install it with:  uv sync   (it ships in the dev group; or:  uv pip install wandb)"
        ) from exc
    return wandb


class WandbRun:
    """A :class:`LoggerRun` backed by one live ``wandb`` run."""

    def __init__(self, wandb_module, run, local_dir: str):
        self._wandb = wandb_module
        self._run = run
        self._dir = local_dir

    def log_param(self, hparam_dict):
        self._run.config.update(dict(hparam_dict), allow_val_change=True)

    def log_metric(self, key, value, step=None):
        # Coerce JAX/NumPy scalars (what the core emits) to a Python float.
        self._run.log({key: float(value)}, step=step)

    def log_histogram(self, key, value, step=None):
        self._run.log({key: self._wandb.Histogram(np.asarray(value))}, step=step)

    def declare_multiline_layout(self, eps):
        # W&B groups metrics by key prefix on its own; it has no custom-scalars
        # layout concept, so the per-epsilon multiline overlay is a no-op here.
        return None

    def get_local_path(self, path):
        return os.path.join(self._dir, path)


class WandbLogger:
    """LoggerFactory + context manager mapping a training run to one W&B run.

    ``__enter__`` is idempotent: the distributed Ray logger actor re-enters per
    call, and every entry returns the same live run, so the centralized actor
    maps to a single W&B run. The run is finished on ``close`` / ``__del__`` (and
    by W&B's own process-exit handler).
    """

    def __init__(
        self,
        wandb_module,
        run_name: str,
        experiment_name: str,
        local_dir: str,
        agent: Optional[Any],
        *,
        entity: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        self._wandb = wandb_module
        self._run_name = run_name
        self._experiment_name = experiment_name
        self._local_dir = os.path.join(local_dir, experiment_name, run_name)
        os.makedirs(self._local_dir, exist_ok=True)
        # experiment_name IS the W&B project (the cross-backend experiment grouping).
        self._project = experiment_name
        self._entity = entity
        self._mode = mode
        self._agent = agent
        self._run = None
        self._logger_run: Optional[WandbRun] = None

    def _ensure_started(self) -> WandbRun:
        if self._logger_run is not None:
            return self._logger_run
        self._run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._run_name,
            dir=self._local_dir,
            mode=self._mode,
        )
        self._logger_run = WandbRun(self._wandb, self._run, self._local_dir)
        if self._agent is not None:
            try:
                self.log_hparams(self._agent)
            except Exception:
                # Match the TensorBoard backend: hparam logging must not break training.
                pass
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

    def __enter__(self) -> WandbRun:
        return self._ensure_started()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Keep the run open across re-entries; it is finished on close()/__del__.
        return False

    def close(self):
        if self._run is not None:
            self._run.finish()
            self._run = None
            self._logger_run = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_wandb_logger_factory(args):
    """Return a ``LoggerFactory`` for W&B, capturing the CLI backend config.

    The ``wandb`` import happens here (only when ``--logger wandb`` is selected),
    so an uninstalled extra fails fast with a clear message before training or
    ``ray.init()`` starts.
    """
    wandb_module = _import_wandb()
    entity = getattr(args, "wandb_entity", None)
    mode = getattr(args, "wandb_mode", None)

    def factory(run_name, experiment_name, local_dir, agent):
        return WandbLogger(
            wandb_module,
            run_name,
            experiment_name,
            local_dir,
            agent,
            entity=entity,
            mode=mode,
        )

    return factory
