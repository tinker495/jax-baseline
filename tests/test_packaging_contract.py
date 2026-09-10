from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_single_distribution_includes_experiment_packages_and_commands():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    project = config["project"]
    setuptools = config["tool"]["setuptools"]
    dependencies = {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        for requirement in project["dependencies"]
    }

    assert not (REPO_ROOT / "adapters").exists()
    assert "workspace" not in config["tool"]["uv"]
    assert set(setuptools["packages"]["find"]["include"]) == {
        "jax_baselines*",
        "env_builder*",
        "experiments*",
        "model_builder*",
        "replay_memory*",
    }
    assert set(project["scripts"]) == {
        "apex-dpg",
        "apex-qnet",
        "dashboard",
        "dpg",
        "exp",
        "impala",
        "pg",
        "qnet",
    }
    for target in project["scripts"].values():
        module, _ = target.split(":")
        assert (REPO_ROOT / (module.replace(".", "/") + ".py")).is_file()
    assert setuptools["package-data"]["experiments"] == ["configs/*.yaml"]
    assert {
        "jax",
        "dm-pix",
        "cpprb",
        "gymnasium",
        "pyyaml",
        "tensorboardx",
    } <= dependencies
    assert "jax-baselines-adapters" not in dependencies
    assert {"all", "mjlab", "distributed", "envpool", "haiku"} <= set(
        project["optional-dependencies"]
    )
    assert set(config["dependency-groups"]["dev"]) == {"pre-commit", "pytest"}
    assert not any(
        item.get("group") == "dev"
        for conflict in config["tool"]["uv"].get("conflicts", [])
        for item in conflict
    )
