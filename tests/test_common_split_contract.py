"""Issue #13 completion contracts for dismantling ``jax_baselines.common``."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MATH_MODULES = (
    "jax_baselines.math.distributional",
    "jax_baselines.math.jax_utils",
    "jax_baselines.math.losses",
    "jax_baselines.math.param_updates",
    "jax_baselines.math.policy_math",
    "jax_baselines.math.returns",
    "jax_baselines.math.schedules",
    "jax_baselines.math.statistics",
)

EXPECTED_CORE_MODULES = (
    "jax_baselines.core.checkpoint",
    "jax_baselines.core.checkpoint_state",
    "jax_baselines.core.env_info",
    "jax_baselines.core.env_protocols",
    "jax_baselines.core.epoch_buffer",
    "jax_baselines.core.eval",
    "jax_baselines.core.hparams",
    "jax_baselines.core.replay_protocol",
    "jax_baselines.core.rollout",
    "jax_baselines.core.rollout_stats",
    "jax_baselines.core.runtime_adapters",
    "jax_baselines.core.serialization",
    "jax_baselines.core.training_session",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_common_package_directory_is_removed_after_split():
    assert not (REPO_ROOT / "jax_baselines" / "common").exists()


def test_math_and_core_modules_are_the_new_import_homes():
    for module_name in EXPECTED_MATH_MODULES + EXPECTED_CORE_MODULES:
        importlib.import_module(module_name)


def test_no_python_imports_pin_deleted_common_paths():
    pinned_imports: list[str] = []
    for root_name in ("jax_baselines", "experiments", "tests"):
        for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            for module in _imported_modules(path):
                if module == "jax_baselines.common" or module.startswith("jax_baselines.common."):
                    pinned_imports.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()} imports {module}"
                    )

    assert not pinned_imports, "deleted common imports remain:\n" + "\n".join(pinned_imports)


def test_representative_symbols_resolve_from_new_homes():
    from jax_baselines.core.checkpoint import CheckpointController
    from jax_baselines.core.env_protocols import EnvInfo
    from jax_baselines.core.replay_protocol import ReplayBufferFactory
    from jax_baselines.core.runtime_adapters import NoOpLogger
    from jax_baselines.core.training_session import TrainingSession
    from jax_baselines.math.distributional import HLGaussTransform
    from jax_baselines.math.losses import QuantileHuberLosses
    from jax_baselines.math.returns import get_vtrace
    from jax_baselines.math.statistics import RunningMeanStd

    assert CheckpointController
    assert EnvInfo
    assert HLGaussTransform
    assert QuantileHuberLosses
    assert NoOpLogger
    assert ReplayBufferFactory
    assert RunningMeanStd
    assert TrainingSession
    assert get_vtrace
