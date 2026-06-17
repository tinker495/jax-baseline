"""`.env`-driven CLI configuration tests (dotenv).

Covers loading a project ``.env`` into ``os.environ`` and the env-derived
defaults of the shared ``--logger`` flags, including precedence:
explicit CLI flag > real shell env var > ``.env`` file > hardcoded default.

``monkeypatch.delenv(..., raising=False)`` is registered before ``load_dotenv``
runs so the env mutations ``load_dotenv`` writes directly to ``os.environ`` are
restored on teardown (kept hermetic).
"""

from __future__ import annotations

import argparse
import os

import pytest

from experiments.cli._common import load_runtime_env
from experiments.cli._loggers import add_logger_args

ENV_VARS = ("JAXBL_LOGGER", "WANDB_ENTITY", "WANDB_MODE", "AIM_REPO")


@pytest.fixture(autouse=True)
def _restore_environ():
    """Restore ``os.environ`` after each test. ``load_dotenv`` writes the real
    environment directly (bypassing monkeypatch), so snapshot/restore here keeps
    these tests from leaking ``JAXBL_LOGGER`` etc. into the rest of the suite."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _parsed(argv):
    parser = argparse.ArgumentParser()
    add_logger_args(parser)
    return parser.parse_args(argv)


def test_load_runtime_env_reads_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "JAXBL_LOGGER=wandb\nWANDB_MODE=offline\nAIM_REPO=/tmp/aimrepo\n"
    )
    monkeypatch.chdir(tmp_path)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    load_runtime_env()

    assert os.environ["JAXBL_LOGGER"] == "wandb"
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["AIM_REPO"] == "/tmp/aimrepo"


def test_load_runtime_env_does_not_override_real_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("JAXBL_LOGGER=aim\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAXBL_LOGGER", "wandb")  # the real shell env must win

    load_runtime_env()

    assert os.environ["JAXBL_LOGGER"] == "wandb"


def test_load_runtime_env_is_a_noop_without_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env on this path
    monkeypatch.delenv("JAXBL_LOGGER", raising=False)

    load_runtime_env()  # must not raise

    assert os.environ.get("JAXBL_LOGGER") is None


def test_logger_flag_defaults_read_environment(monkeypatch):
    monkeypatch.setenv("JAXBL_LOGGER", "wandb")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("AIM_REPO", "/repo/aim")

    args = _parsed([])

    assert args.logger == "wandb"
    assert args.wandb_entity == "team"
    assert args.wandb_mode == "offline"
    assert args.aim_repo == "/repo/aim"


def test_explicit_flags_override_environment(monkeypatch):
    monkeypatch.setenv("JAXBL_LOGGER", "wandb")
    monkeypatch.setenv("WANDB_MODE", "online")

    args = _parsed(["--logger", "aim", "--wandb_mode", "offline"])

    assert args.logger == "aim"
    assert args.wandb_mode == "offline"


def test_logger_default_is_tensorboard_without_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    args = _parsed([])

    assert args.logger == "tensorboard"
    assert args.wandb_entity is None
    assert args.aim_repo is None
