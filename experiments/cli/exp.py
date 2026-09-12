"""Config-driven experiment sweep runner (``exp``).

Reads a YAML sweep definition and launches each variant as an isolated
subprocess invoking the selected runner console script. Process-per-variant
preserves the JAX/XLA + GPU-memory isolation the legacy shell scripts relied on.

YAML schema
-----------
category: experiment topic, e.g. atari_100k | atari | mjlab | mujoco
runner:   one of qnet | dpg | pg | impala | apex_qnet | apex_dpg
base:     mapping of CLI args shared by every variant (keys without leading --)
variants: list of mappings; each is merged over ``base`` (variant wins).
          A variant may set ``enabled: false`` to keep it on record but skipped.
runtime:  optional mapping:
            device: value for CUDA_VISIBLE_DEVICES
            xvfb:   bool, wrap each command with ``xvfb-run -a``

Argument encoding:
  bool true  -> ``--key``        (store_true flag)
  bool false -> omitted
  other      -> ``--key value``

CLI overrides:
  ``--set KEY=VALUE`` (repeatable) overrides an arg for every variant, applied on
  top of base/variant. Useful to cap memory per machine, e.g. --set buffer_size=1e5.

Network files:
  ``model`` (Q-Net), ``actor_model`` and ``critic_model`` are JSON paths relative
  to the YAML file, including paths supplied through ``--set``. ``--dry-run``
  validates and prints their definitions. ``--export-models DIR`` writes the
  normalized definitions and uses those files in the generated commands.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.cli._common import load_runtime_env
from model_builder.model_config import load_model_config, model_config_dict

RUNNER_SCRIPTS = {
    "qnet": "qnet",
    "dpg": "dpg",
    "pg": "pg",
    "impala": "impala",
    "apex_qnet": "apex-qnet",
    "apex_dpg": "apex-dpg",
}
MODEL_ARGUMENTS = ("model", "actor_model", "critic_model")


def _build_args(merged):
    argv = []
    for key, value in merged.items():
        if value is False:
            continue
        argv.append(f"--{key}")
        if value is not True:
            argv.append(str(value))
    return argv


def _iter_commands(config, cli_overrides=None, *, config_dir: Path | None = None):
    cli_overrides = cli_overrides or {}
    if config_dir is None:
        config_dir = Path.cwd()
    runner = config["runner"]
    if not isinstance(runner, str) or runner not in RUNNER_SCRIPTS:
        raise ValueError(f"unknown runner '{runner}', expected one of {sorted(RUNNER_SCRIPTS)}")
    script = RUNNER_SCRIPTS[runner]
    base = config.get("base") or {}
    runtime = config.get("runtime") or {}
    xvfb = runtime.get("xvfb", False)
    for variant in config.get("variants") or [{}]:
        variant = variant or {}
        if not variant.get("enabled", True):
            continue
        variant_args = {k: v for k, v in variant.items() if k != "enabled"}
        merged = {**base, **variant_args, **cli_overrides}
        for name in MODEL_ARGUMENTS:
            if name not in merged:
                continue
            if not isinstance(merged[name], str) or not merged[name]:
                raise ValueError(f"{name} must be a nonempty JSON file path")
            merged[name] = str((config_dir / merged[name]).resolve())
        command = [script, *_build_args(merged)]
        if xvfb:
            command = ["xvfb-run", "-a", *command]
        yield command


def main(argv=None):
    # Behave like a normal Unix tool when piped into head/grep (no BrokenPipe traceback).
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    # Load .env before snapshotting os.environ so spawned variant subprocesses
    # inherit JAXBL_/WANDB_/AIM_ settings (each child also loads it via run_family).
    load_runtime_env()
    parser = argparse.ArgumentParser(description="Run a YAML-defined experiment sweep.")
    parser.add_argument("config", help="path to a sweep YAML file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate model JSON and print commands without running them",
    )
    parser.add_argument(
        "--export-models",
        type=Path,
        metavar="DIR",
        help="write normalized model JSON per variant and use the exported files",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an arg for every variant (repeatable), e.g. --set buffer_size=1e5",
    )
    args = parser.parse_args(argv)

    cli_overrides = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        cli_overrides[key] = yaml.safe_load(value)

    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)

    category = config["category"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a nonempty string")

    runtime = config.get("runtime") or {}
    env = os.environ.copy()
    if "device" in runtime:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["device"])

    commands = list(_iter_commands(config, cli_overrides, config_dir=Path(args.config).parent))
    model_definitions = [
        {
            name: model_config_dict(load_model_config(command[command.index(f"--{name}") + 1]))
            for name in MODEL_ARGUMENTS
            if f"--{name}" in command
        }
        for command in commands
    ]
    export_dir = args.export_models.resolve() if args.export_models is not None else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
    print(f"category: {category}", flush=True)
    failures = []
    for index, (command, models) in enumerate(
        zip(commands, model_definitions, strict=True), start=1
    ):
        if export_dir is not None:
            for name, definition in models.items():
                destination = export_dir / f"{index:03d}-{name}.json"
                destination.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
                command[command.index(f"--{name}") + 1] = str(destination)
        printable = shlex.join(command)
        print(f"[{index}/{len(commands)}] {printable}", flush=True)
        if args.dry_run:
            if models:
                print(json.dumps(models, indent=2), flush=True)
            continue
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            failures.append((index, printable, result.returncode))

    if failures:
        print(f"\n{len(failures)} variant(s) failed:", file=sys.stderr)
        for index, printable, code in failures:
            print(f"  [{index}] exit {code}: {printable}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
