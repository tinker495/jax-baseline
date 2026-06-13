"""Regression test: APE-X ``worker.run()`` must be called with eps/seed by keyword.

``worker.run`` / ``dpg_worker.run`` end with ``(..., eps=0.05, seed=None)``. A prior
version passed the last two positionally as ``(self.seed + idx, eps)`` — swapping
them, so each actor's exploration epsilon became ``seed + idx`` and the PRNG was
seeded with the intended epsilon. Lock both call sites to keyword form so the swap
cannot silently return. Parsed via AST to avoid importing the distributed stack.
"""

import ast
from pathlib import Path

import pytest

CALL_SITES = [
    "jax_baselines/APE_X/base_class.py",
    "jax_baselines/APE_X/dpg_base_class.py",
]


def _find_run_remote_call(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "remote"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "run"
        ):
            return node
    return None


@pytest.mark.parametrize("relative_path", CALL_SITES)
def test_apex_run_passes_eps_and_seed_by_keyword(relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / relative_path).read_text())

    call = _find_run_remote_call(tree)
    assert call is not None, f"no worker.run.remote(...) call found in {relative_path}"

    keyword_args = {kw.arg for kw in call.keywords}
    assert "eps" in keyword_args, f"{relative_path}: run.remote must pass eps= by keyword"
    assert "seed" in keyword_args, f"{relative_path}: run.remote must pass seed= by keyword"
