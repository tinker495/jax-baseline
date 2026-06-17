"""Shared pytest fixtures.

Keep the suite deterministic regardless of a developer's repo ``.env`` or
exported ``JAXBL_*`` / ``WANDB_*`` / ``AIM_*`` variables. The CLI dispatch entry
points (``run_family`` / ``run_distributed_family`` / ``dashboard`` / ``exp``)
call ``load_runtime_env`` on startup, which would otherwise pull a local ``.env``
into tests of default behavior. Neutralize that loader at those entry points and
clear the logging env vars. The loader itself is tested directly in
``test_cli_env_config.py`` (via its own tmp-dir ``.env``), which is unaffected
because it imports ``load_runtime_env`` from ``experiments.cli._common``.
"""

import pytest

_DISPATCH_MODULES = (
    "experiments.cli._run",
    "experiments.cli.dashboard",
    "experiments.cli.exp",
)
_LOGGING_ENV = (
    "JAXBL_LOGGER",
    "JAXBL_LOG_DIR",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "AIM_REPO",
)


@pytest.fixture(autouse=True)
def _isolate_runtime_env(monkeypatch):
    for var in _LOGGING_ENV:
        monkeypatch.delenv(var, raising=False)
    for module in _DISPATCH_MODULES:
        monkeypatch.setattr(f"{module}.load_runtime_env", lambda: None, raising=False)
