"""Shared helpers for jax_baselines command-line entry points."""

import os

from dotenv import find_dotenv, load_dotenv

LOG_DIR_ENV = "JAXBL_LOG_DIR"
LOGGER_ENV = "JAXBL_LOGGER"

XLA_FLAGS = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "


def load_runtime_env() -> None:
    """Load a project ``.env`` (if present) into ``os.environ``.

    Lets runtime configuration — logging backend (``JAXBL_LOGGER``), log
    directory (``JAXBL_LOG_DIR``), and W&B / Aim credentials and settings
    (``WANDB_API_KEY``, ``WANDB_PROJECT``, ``WANDB_ENTITY``, ``WANDB_MODE``,
    ``AIM_REPO``) — live in a file instead of the shell. The search walks up
    from the current working directory, so runners launched from any directory
    pick up the repo ``.env``. Real environment variables already set take
    precedence (``override=False``), so the shell still wins over the file.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def set_default_xla_flags() -> None:
    """Apply the shared XLA GPU flags used by every runner entry point."""
    os.environ["XLA_FLAGS"] = XLA_FLAGS


def default_logdir(family: str) -> str:
    """Resolve the default log directory for a runner family.

    Anchored to ``$JAXBL_LOG_DIR`` (default ``runs``) under the current working
    directory, so runners no longer need to be launched from inside ``test/``.
    Always overridable with ``--logdir``.
    """
    root = os.environ.get(LOG_DIR_ENV, "runs")
    return os.path.join(root, family)


set_default_xla_flags()
