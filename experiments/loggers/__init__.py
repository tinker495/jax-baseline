"""Concrete logging-adapter backends selectable via ``--logger``.

Each backend satisfies :class:`jax_baselines.core.runtime_adapters.LoggerRun`.
Third-party deps (wandb, aim) are optional dependencies imported lazily inside
the backend module so the default tensorboard backend needs neither installed.
"""
