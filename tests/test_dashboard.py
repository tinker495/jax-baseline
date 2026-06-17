"""``dashboard`` viewer-launcher tests.

``dashboard`` ends in ``os.execvp`` (it replaces the process with the viewer), so
the tests monkeypatch ``shutil.which`` + ``os.execvp`` to capture the command
that *would* run instead of actually launching a server.
"""

from __future__ import annotations

import os

import pytest

from experiments.cli import dashboard

ENV_VARS = ("JAXBL_LOGGER", "JAXBL_LOG_DIR", "AIM_REPO")


class _Exec(Exception):
    """Raised in place of os.execvp so main() stops at the exec point."""


@pytest.fixture(autouse=True)
def _capture(monkeypatch):
    """Capture the exec'd command and keep the env hermetic (load_runtime_env
    writes os.environ directly via dotenv)."""
    saved = dict(os.environ)
    for var in ENV_VARS:
        os.environ.pop(var, None)
    rec: dict = {}

    def fake_execvp(exe, argv):
        rec["exe"] = exe
        rec["argv"] = list(argv)
        raise _Exec

    # Pretend the viewer packages are installed in this environment.
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *a, **k: object() if name in ("tensorboard", "aim") else None,
    )
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr("os.execvp", fake_execvp)
    # Don't create a real .aim repo on disk during tests; record the path instead.
    monkeypatch.setattr(
        "experiments.cli.dashboard._ensure_aim_repo",
        lambda repo: rec.__setitem__("aim_repo_init", repo),
    )
    yield rec
    os.environ.clear()
    os.environ.update(saved)


def test_tensorboard_command(_capture):
    with pytest.raises(_Exec):
        dashboard.main(
            ["--logger", "tensorboard", "--logdir", "runs", "--host", "0.0.0.0", "--port", "6006"]
        )
    assert _capture["argv"] == [
        "/usr/bin/tensorboard",
        "--logdir",
        "runs",
        "--host",
        "0.0.0.0",
        "--port",
        "6006",
    ]


def test_aim_command(_capture):
    with pytest.raises(_Exec):
        dashboard.main(["--logger", "aim", "--aim_repo", "./.aim", "--host", "127.0.0.1"])
    assert _capture["argv"] == ["/usr/bin/aim", "up", "--repo", "./.aim", "--host", "127.0.0.1"]


def test_aim_defaults_repo_to_dot_aim(_capture):
    with pytest.raises(_Exec):
        dashboard.main(["--logger", "aim"])
    assert _capture["argv"][:4] == ["/usr/bin/aim", "up", "--repo", ".aim"]


def test_aim_initializes_repo_before_launch(_capture):
    # `aim up` errors on a missing repo; the launcher initializes it first.
    with pytest.raises(_Exec):
        dashboard.main(["--logger", "aim", "--aim_repo", "myrepo"])
    assert _capture["aim_repo_init"] == "myrepo"


def test_tensorboard_does_not_init_aim_repo(_capture):
    with pytest.raises(_Exec):
        dashboard.main(["--logger", "tensorboard"])
    assert "aim_repo_init" not in _capture


def test_env_selects_backend(_capture, monkeypatch):
    monkeypatch.setenv("JAXBL_LOGGER", "aim")
    with pytest.raises(_Exec):
        dashboard.main([])
    assert _capture["argv"][0] == "/usr/bin/aim"


def test_default_is_tensorboard(_capture):
    with pytest.raises(_Exec):
        dashboard.main([])
    assert _capture["argv"][0] == "/usr/bin/tensorboard"


def test_unknown_args_forwarded_to_viewer(_capture):
    with pytest.raises(_Exec):
        dashboard.main(["--logger", "tensorboard", "--reload_interval", "5"])
    assert _capture["argv"][-2:] == ["--reload_interval", "5"]


def test_uninstalled_viewer_raises_actionable_error(_capture, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    with pytest.raises(SystemExit, match="tensorboard.*not installed"):
        dashboard.main(["--logger", "tensorboard"])


def test_installed_but_cli_not_on_path_errors(_capture, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)  # importable, but no script
    with pytest.raises(SystemExit, match="not on PATH"):
        dashboard.main(["--logger", "aim"])


def test_wandb_prints_guidance_without_exec(_capture, capsys):
    assert dashboard.main(["--logger", "wandb"]) == 0
    out = capsys.readouterr().out
    assert "wandb sync" in out
    assert "exe" not in _capture  # no viewer was exec'd
