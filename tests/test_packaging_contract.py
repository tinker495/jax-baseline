from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _project(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        for requirement in requirements
    }


def test_core_and_adapter_distribution_boundaries():
    core = _project(REPO_ROOT / "pyproject.toml")
    adapters = _project(REPO_ROOT / "adapters" / "pyproject.toml")
    adapter_dependencies = _requirement_names(adapters["project"]["dependencies"])
    adapter_extras = adapters["project"]["optional-dependencies"]

    assert core["tool"]["setuptools"]["packages"]["find"]["include"] == ["jax_baselines*"]
    assert "scripts" not in core["project"]
    assert _requirement_names(core["project"]["dependencies"]) == {
        "chex",
        "dm-pix",
        "flax",
        "jax",
        "numpy",
        "optax",
    }
    assert core["tool"]["uv"]["workspace"]["members"] == ["adapters"]
    assert core["dependency-groups"]["dev"][0] == "jax-baselines-adapters[all]"

    assert set(adapters["project"]["scripts"]) == {
        "apex-dpg",
        "apex-qnet",
        "dashboard",
        "dpg",
        "exp",
        "impala",
        "pg",
        "qnet",
    }
    assert adapters["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "env_builder*",
        "experiments*",
        "model_builder*",
        "replay_memory*",
    ]
    assert adapters["tool"]["setuptools"]["package-data"]["experiments"] == ["configs/*.yaml"]
    assert {"all", "distributed", "envpool", "wandb", "aim"} <= set(adapter_extras)
    assert {
        "aim",
        "autorom",
        "box2d-kengz",
        "dm-haiku",
        "envpool",
        "opencv-python",
        "ray",
        "tensorboard",
        "wandb",
    }.isdisjoint(adapter_dependencies)
    assert {"cpprb", "gymnasium", "python-dotenv", "pyyaml", "tensorboardx", "tqdm"} <= (
        adapter_dependencies
    )


def test_adapter_build_tree_reuses_repository_packages():
    source = REPO_ROOT / "adapters" / "src"
    for package in ("env_builder", "experiments", "model_builder", "replay_memory"):
        link = source / package
        assert link.is_symlink()
        assert link.resolve() == REPO_ROOT / package
