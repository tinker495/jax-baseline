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
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.cli._common import default_logdir, load_runtime_env
from experiments.cli._validation import parse_runner_args, runner_parser
from experiments.cli.apex_dpg import APEX_DPG_RUNNER
from experiments.cli.apex_qnet import APEX_QNET_RUNNER
from experiments.cli.dpg import DPG_RUNNER
from experiments.cli.impala import IMPALA_RUNNER
from experiments.cli.pg import PG_RUNNER
from experiments.cli.qnet import QNET_RUNNER
from experiments.run_metadata import SWEEP_CONTEXT_ENV
from model_builder.model_config import load_model_config, model_config_dict

RUNNERS = {
    "qnet": QNET_RUNNER,
    "dpg": DPG_RUNNER,
    "pg": PG_RUNNER,
    "impala": IMPALA_RUNNER,
    "apex_qnet": APEX_QNET_RUNNER,
    "apex_dpg": APEX_DPG_RUNNER,
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
    if not isinstance(config, dict) or "runner" not in config or "category" not in config:
        raise ValueError("sweep must be a mapping with a runner and category")
    category = config["category"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a nonempty string")
    config = {"base": {}, "runtime": {}, "variants": [{}], **config}
    runner = config["runner"]
    if not isinstance(runner, str) or runner not in RUNNERS:
        raise ValueError(f"unknown runner '{runner}', expected one of {sorted(RUNNERS)}")
    script = runner.replace("_", "-")
    parser = runner_parser(RUNNERS[runner], prog=script)
    base = {} if config["base"] is None else config["base"]
    runtime = {} if config["runtime"] is None else config["runtime"]
    variants = (
        [{}] if config["variants"] is None or config["variants"] == [] else config["variants"]
    )
    if not isinstance(base, dict) or not isinstance(runtime, dict):
        raise TypeError("base and runtime must be mappings")
    if not isinstance(variants, list):
        raise TypeError("variants must be a list of argument mappings")
    runtime = {"xvfb": False, **runtime}
    if not isinstance(runtime["xvfb"], bool):
        raise TypeError("runtime.xvfb must be a boolean")
    for variant in variants:
        variant = {} if variant is None else variant
        if not isinstance(variant, dict):
            raise TypeError("each variant must be an argument mapping")
        variant = {"enabled": True, **variant}
        if not isinstance(variant["enabled"], bool):
            raise TypeError("variant.enabled must be a boolean")
        if not variant["enabled"]:
            continue
        variant_args = {k: v for k, v in variant.items() if k != "enabled"}
        merged = {
            "logdir": default_logdir(category),
            **base,
            **variant_args,
            **cli_overrides,
        }
        for key, value in merged.items():
            if f"--{key}" not in parser._option_string_actions:
                parser.error(f"unrecognized argument: --{key}")
            if isinstance(value, bool) and not isinstance(
                parser._option_string_actions[f"--{key}"], argparse._StoreConstAction
            ):
                parser.error(f"--{key} requires a value, not a boolean")
            if not isinstance(value, (str, int, float, bool)):
                parser.error(f"--{key} requires a scalar value")
        for name in MODEL_ARGUMENTS:
            if name not in merged:
                continue
            model_path = merged[name]
            if not isinstance(model_path, str) or not model_path:
                raise ValueError(f"{name} must be a nonempty JSON file path")
            merged[name] = str((config_dir / model_path).resolve())
        variant_argv = _build_args(merged)
        parse_runner_args(RUNNERS[runner], variant_argv, prog=script)
        command = [script, *variant_argv]
        if runtime["xvfb"]:
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
        help="validate all runner arguments, algorithm/backend combinations and model JSON",
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

    try:
        source = Path(args.config).read_text(encoding="utf-8")
        config = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    try:
        commands = list(_iter_commands(config, cli_overrides, config_dir=Path(args.config).parent))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    category = config["category"]
    runtime = config["runtime"] if "runtime" in config and config["runtime"] is not None else {}
    env = os.environ.copy()
    if "device" in runtime:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["device"])

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
        env[SWEEP_CONTEXT_ENV] = json.dumps(
            {
                "config_path": str(Path(args.config).resolve()),
                "config_yaml": source,
                "config_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "category": category,
                "runner": config["runner"],
                "enabled_variant_index": index,
                "overrides": cli_overrides,
            }
        )
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
