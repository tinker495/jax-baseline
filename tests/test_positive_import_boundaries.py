"""Positive-direction import test for the adapter-free Algorithm Core.

The static ratchet in ``test_import_boundaries.py`` catches direct forbidden
source imports.  This integration-shaped check executes module top levels in a
fresh subprocess while concrete adapter dependencies are made unavailable,
which catches transitive coupling that a source scan intentionally cannot see.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKED_ADAPTER_ROOTS = (
    "cpprb",
    "env_builder",
    "envpool",
    "experiments",
    "gymnasium",
    "model_builder",
    "ray",
    "replay_memory",
    "tensorboardX",
)


def test_jax_baselines_modules_import_without_concrete_adapters():
    script = f"""
from __future__ import annotations

import importlib
import importlib.abc
import pkgutil
import sys
import traceback

blocked = {BLOCKED_ADAPTER_ROOTS!r}


class BlockedAdapterFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(tuple(root + "." for root in blocked)):
            raise ModuleNotFoundError(
                f"blocked optional adapter dependency for positive import test: {{fullname}}"
            )
        return None


sys.meta_path.insert(0, BlockedAdapterFinder())

failures = []
root = importlib.import_module("jax_baselines")
for module_info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
    try:
        importlib.import_module(module_info.name)
    except Exception as exc:  # pragma: no cover - reported to parent process.
        reason = traceback.format_exception_only(type(exc), exc)[-1].strip()
        failures.append((module_info.name, reason))

if failures:
    print("Concrete adapter import failures:")
    for name, reason in failures:
        print(f"  - {{name}}: {{reason}}")
    raise SystemExit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
