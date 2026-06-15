"""Shared helpers for jax_baselines command-line entry points."""

import os

LOG_DIR_ENV = "JAXBL_LOG_DIR"

XLA_FLAGS = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "


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
