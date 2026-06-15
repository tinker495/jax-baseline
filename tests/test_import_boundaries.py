"""Static import-boundary ratchet for the ``jax_baselines`` core package.

ADR 0002 makes this test intentionally stale-aware: the live set of forbidden
imports under ``jax_baselines/`` must match the declared allowlist exactly. A
new forbidden import is an undeclared violation; removing a forbidden import
without retiring its allowlist entry is a stale exemption. Both conditions fail
so migration slices shrink the allowlist in the same changeset that moves code.

The scanner parses source only. It never imports ``jax_baselines`` or optional
runtime adapters such as Gymnasium, EnvPool, Ray, cpprb, or TensorBoard.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = REPO_ROOT / "jax_baselines"
RUNTIME_NEUTRAL_ROOTS = (
    REPO_ROOT / "jax_baselines" / "core",
    REPO_ROOT / "jax_baselines" / "math",
)

FORBIDDEN_TOKENS = (
    "gymnasium",
    "envpool",
    "cpprb",
    "ray",
    "argparse",
    "tensorboardX",
    "model_builder",
    "env_builder",
    "replay_memory",
    "experiments",
)
ALLOWLIST_TAGS = {"NOT_YET_MOVED", "MIGRATION_SHIM", "DISTRIBUTED_RUNTIME_EXEMPTION"}


@dataclass(frozen=True)
class AllowedImport:
    path: str
    token: str
    tag: str
    retiring_slice: str

    @property
    def key(self) -> tuple[str, str]:
        return self.path, self.token


# Issue #13 finalized the migration ledger: the core has no static imports of
# concrete adapter packages.  Future exceptions must be added deliberately and
# will be checked by the tests below.
ALLOWED_IMPORTS: tuple[AllowedImport, ...] = ()


def _iter_python_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _module_names_from_import(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        return (alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return (node.module,)
        return (alias.name for alias in node.names)
    return ()


def _matching_forbidden_tokens(module_name: str) -> Iterable[str]:
    """Yield ADR 0002 forbidden tokens for a direct imported module name.

    ADR 0002 defines a token ``T`` as matching module ``M`` only when
    ``M == T`` or ``M.startswith(T + ".")``.  That intentionally targets
    direct imports of concrete adapter packages and future sibling packages
    such as ``env_builder``; it does not suffix-match same-package imports like
    ``env_builder.env_builder``.  Transitive coupling through those
    core modules is explicitly deferred by the ADR.
    """

    for token in FORBIDDEN_TOKENS:
        if module_name == token or module_name.startswith(f"{token}."):
            yield token


def _find_forbidden_imports(root: Path) -> set[tuple[str, str]]:
    forbidden_imports: set[tuple[str, str]] = set()
    for path in _iter_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            for module_name in _module_names_from_import(node):
                for token in _matching_forbidden_tokens(module_name):
                    forbidden_imports.add((relative_path, token))
    return forbidden_imports


def _find_forbidden_imports_in_source(path: str, source: str) -> set[tuple[str, str]]:
    tree = ast.parse(source, filename=path)
    forbidden_imports: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        for module_name in _module_names_from_import(node):
            for token in _matching_forbidden_tokens(module_name):
                forbidden_imports.add((path, token))
    return forbidden_imports


def _literal_import_target(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
    return None


def _is_dynamic_import_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in {"import_module", "__import__"}
    return isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"


def _find_forbidden_dynamic_imports(root: Path) -> set[tuple[str, str]]:
    forbidden_imports: set[tuple[str, str]] = set()
    for path in _iter_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not _is_dynamic_import_call(node):
                continue
            module_name = _literal_import_target(node)
            if module_name is None:
                continue
            for token in _matching_forbidden_tokens(module_name):
                forbidden_imports.add((relative_path, token))
    return forbidden_imports


def _find_forbidden_dynamic_imports_in_source(path: str, source: str) -> set[tuple[str, str]]:
    tree = ast.parse(source, filename=path)
    forbidden_imports: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not _is_dynamic_import_call(node):
            continue
        module_name = _literal_import_target(node)
        if module_name is None:
            continue
        for token in _matching_forbidden_tokens(module_name):
            forbidden_imports.add((path, token))
    return forbidden_imports


def _allowlist_keys() -> set[tuple[str, str]]:
    return {entry.key for entry in ALLOWED_IMPORTS}


def _format_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    pairs = sorted(pairs)
    if not pairs:
        return "  (none)"
    return "\n".join(f"  - {path} :: {token}" for path, token in pairs)


def _boundary_failure_message(undeclared: set[tuple[str, str]], stale: set[tuple[str, str]]) -> str:
    return (
        "Import-boundary allowlist drift detected.\n\n"
        "Undeclared forbidden imports (actual - declared):\n"
        f"{_format_pairs(undeclared)}\n\n"
        "Stale allowlist entries (declared - actual):\n"
        f"{_format_pairs(stale)}\n\n"
        "If this is a new not-yet-moved adapter import, declare it with a tag and retiring slice. "
        "If a migration removed an import, delete the stale allowlist entry in the same changeset. "
        "See docs/adr/0002-enforce-import-boundaries-with-stale-aware-allowlist-ratchet.md."
    )


def _assert_allowlist_matches(actual: set[tuple[str, str]], declared: set[tuple[str, str]]) -> None:
    undeclared = actual - declared
    stale = declared - actual

    assert actual == declared, _boundary_failure_message(undeclared, stale)


def test_import_boundary_allowlist_matches_current_forbidden_imports():
    actual = _find_forbidden_imports(SCAN_ROOT)
    declared = _allowlist_keys()

    _assert_allowlist_matches(actual, declared)


def test_runtime_neutral_core_and_math_have_no_dynamic_adapter_imports():
    actual = set().union(*(_find_forbidden_dynamic_imports(root) for root in RUNTIME_NEUTRAL_ROOTS))

    assert actual == set(), _boundary_failure_message(actual, set())


def test_dynamic_import_scanner_detects_literal_adapter_imports():
    source = """
import importlib
from importlib import import_module

import_module("ray")
importlib.import_module("gymnasium.wrappers")
__import__("cpprb")
import_module(dynamic_name)
"""

    assert _find_forbidden_dynamic_imports_in_source("synthetic.py", source) == {
        ("synthetic.py", "cpprb"),
        ("synthetic.py", "gymnasium"),
        ("synthetic.py", "ray"),
    }


def test_import_boundary_scanner_detects_direct_nested_type_checking_and_duplicates():
    source = """
import gymnasium as gym
import gymnasium
import gymnasium_extra
import math
import ray
from cpprb import ReplayBuffer
from gymnasium.wrappers import AtariPreprocessing
from mypkg import ray_helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import model_builder


def build():
    import envpool
    from experiments.sweeps import runner
"""

    assert _find_forbidden_imports_in_source("synthetic.py", source) == {
        ("synthetic.py", "cpprb"),
        ("synthetic.py", "envpool"),
        ("synthetic.py", "experiments"),
        ("synthetic.py", "gymnasium"),
        ("synthetic.py", "model_builder"),
        ("synthetic.py", "ray"),
    }


def test_import_boundary_scanner_preserves_direct_top_level_token_semantics():
    source = """
from env_builder.env_builder import get_env
from jax_baselines.core.env_builder import get_core_env
from jax_baselines.core.cpprb_buffers import ReplayBuffer
from jax_baselines.core import env_info
from env_builder.factory import make_env
from replay_memory.buffers import ReplayBuffer as AdapterReplayBuffer
"""

    assert _find_forbidden_imports_in_source("synthetic.py", source) == {
        ("synthetic.py", "env_builder"),
        ("synthetic.py", "replay_memory"),
    }


def test_import_boundary_scanner_detects_relative_importfrom_aliases():
    source = """
from . import env_builder
from .. import replay_memory as replay
from . import gymnasium_extra
"""

    assert _find_forbidden_imports_in_source("synthetic.py", source) == {
        ("synthetic.py", "env_builder"),
        ("synthetic.py", "replay_memory"),
    }


def test_import_boundary_allowlist_entries_are_documented():
    seen_keys: set[tuple[str, str]] = set()
    duplicate_keys: set[tuple[str, str]] = set()

    for entry in ALLOWED_IMPORTS:
        if entry.key in seen_keys:
            duplicate_keys.add(entry.key)
        seen_keys.add(entry.key)

        assert entry.path.endswith(".py"), f"allowlist path must be a Python file: {entry}"
        assert entry.token in FORBIDDEN_TOKENS, f"unknown allowlist token: {entry}"
        assert entry.tag in ALLOWLIST_TAGS, f"unknown allowlist tag: {entry}"
        assert entry.retiring_slice.strip(), f"missing retiring-slice note: {entry}"

    assert not duplicate_keys, "duplicate allowlist entries:\n" + _format_pairs(duplicate_keys)


def test_import_boundary_final_slice_has_no_common_or_migration_debt_allowlist_entries():
    """Issue #13 is the final ``common`` dismantling slice.

    The allowlist may be non-empty only for consciously recorded distributed
    runtime wiring exemptions.  Migration shims, not-yet-moved entries, and
    anything under ``jax_baselines/common`` should be gone in the same
    changeset that seals the boundary.
    """

    migration_debt = [
        entry
        for entry in ALLOWED_IMPORTS
        if entry.tag != "DISTRIBUTED_RUNTIME_EXEMPTION"
        or entry.path.startswith("jax_baselines/common/")
    ]

    assert not migration_debt, (
        "final boundary allowlist must contain only explicit distributed runtime exemptions:\n"
        + "\n".join(
            f"  - {entry.path} :: {entry.token} [{entry.tag}] {entry.retiring_slice}"
            for entry in migration_debt
        )
    )


def test_import_boundary_remaining_exemptions_are_distributed_ray_only():
    for entry in ALLOWED_IMPORTS:
        assert entry.tag == "DISTRIBUTED_RUNTIME_EXEMPTION"
        assert entry.token == "ray"
        assert "distributed" in entry.retiring_slice.lower()


def test_import_boundary_failure_message_distinguishes_undeclared_and_stale_entries():
    actual = {("jax_baselines/new_runtime_leak.py", "experiments")}
    declared = {("jax_baselines/removed_runtime_leak.py", "ray")}

    try:
        _assert_allowlist_matches(actual, declared)
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected allowlist drift to fail")

    assert "Undeclared forbidden imports (actual - declared):" in message
    assert "  - jax_baselines/new_runtime_leak.py :: experiments" in message
    assert "Stale allowlist entries (declared - actual):" in message
    assert "  - jax_baselines/removed_runtime_leak.py :: ray" in message
