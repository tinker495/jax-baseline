"""``dashboard`` — launch the configured logging backend's local viewer.

Selects the backend the same way the runners do (``--logger``, defaulting to
``$JAXBL_LOGGER`` / ``tensorboard``, with ``.env`` honored) and starts its
dashboard server:

- ``tensorboard`` -> ``tensorboard --logdir <logdir> --host <host> [--port ...]``
- ``aim``         -> ``aim up --repo <repo> --host <host> [--port ...]``
- ``wandb``       -> cloud-hosted; prints how to view online / sync offline runs.

Unknown arguments are forwarded to the underlying viewer, e.g.
``uv run dashboard --port 7000 --reload_interval 5``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil

from experiments.cli._common import LOG_DIR_ENV, LOGGER_ENV, load_runtime_env
from experiments.cli._loggers import DEFAULT_LOGGER, LOGGER_BACKENDS

# How to install each viewer if it is missing from the current environment.
# These ship in the dev dependency group, so `uv sync` is the usual fix.
_INSTALL_HINT = {
    "tensorboard": "uv sync   (installs it via the dev group; or: uv pip install 'tensorboard' 'setuptools<81')",
    "aim": "uv sync   (installs it via the dev group; or: uv pip install aim)",
}


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logger",
        type=str,
        default=os.environ.get(LOGGER_ENV, DEFAULT_LOGGER),
        choices=list(LOGGER_BACKENDS),
        help=f"backend whose dashboard to launch (default tensorboard; env ${LOGGER_ENV})",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.environ.get(LOG_DIR_ENV, "runs"),
        help=f"TensorBoard log root (default 'runs'; env ${LOG_DIR_ENV})",
    )
    parser.add_argument(
        "--aim_repo",
        type=str,
        default=os.environ.get("AIM_REPO"),
        help="Aim repo path (env $AIM_REPO; default ./.aim)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host/interface to bind (default 0.0.0.0 — reachable remotely)",
    )
    parser.add_argument("--port", type=int, default=None, help="port (default: the viewer's own)")


def _ensure_aim_repo(repo: str) -> None:
    """`aim up` errors if the repo does not exist yet (before any ``--logger aim``
    run has created it). Initialize an empty repo at the same path training uses,
    so the dashboard opens (empty until runs exist). Idempotent on an existing repo.
    """
    import aim

    aim.Repo.from_path(repo, init=True)


def _launch(package: str, cmd: list[str], prepare=None) -> None:
    """Replace this process with the viewer so Ctrl-C maps straight to it.

    Gated on the viewer being importable in *this* environment, so we never exec
    a stray (and possibly broken) CLI from an unrelated Python on ``PATH``; under
    ``uv run`` the venv's own console script is first on ``PATH`` once installed.
    ``prepare`` runs after that check (so it may import the viewer package).
    """
    if importlib.util.find_spec(package) is None:
        raise SystemExit(
            f"the '{package}' viewer is not installed in this environment. Install it with:\n"
            f"    {_INSTALL_HINT[package]}"
        )
    if prepare is not None:
        prepare()
    exe = shutil.which(cmd[0])
    if exe is None:
        raise SystemExit(f"'{package}' is installed but its '{cmd[0]}' CLI is not on PATH.")
    os.execvp(exe, [exe, *cmd[1:]])


def _wandb_guidance(logdir: str) -> int:
    print(
        "W&B is cloud-hosted — there is no local dashboard server to start.\n"
        "  online runs : open the run URL printed during training "
        "(https://wandb.ai/<entity>/<project>)\n"
        f"  offline runs: upload them with  wandb sync {os.path.join(logdir, 'wandb', 'offline-run-*')}"
    )
    return 0


def main(argv=None) -> int:
    load_runtime_env()
    parser = argparse.ArgumentParser(
        prog="dashboard", description="Launch the configured logger's dashboard."
    )
    _add_args(parser)
    args, extra = parser.parse_known_args(argv)

    if args.logger == "wandb":
        return _wandb_guidance(args.logdir)

    prepare = None
    if args.logger == "tensorboard":
        package = "tensorboard"
        cmd = ["tensorboard", "--logdir", args.logdir, "--host", args.host]
    else:  # aim
        package = "aim"
        repo = args.aim_repo or ".aim"
        cmd = ["aim", "up", "--repo", repo, "--host", args.host]

        def prepare():
            _ensure_aim_repo(repo)

    if args.port is not None:
        cmd += ["--port", str(args.port)]
    cmd += extra

    _launch(package, cmd, prepare)
    return 0  # unreachable (execvp replaces the process) unless the viewer is missing


if __name__ == "__main__":
    raise SystemExit(main())
