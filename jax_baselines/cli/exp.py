"""Config-driven experiment sweep runner (``exp``).

Reads a YAML sweep definition and launches each variant as an isolated
subprocess invoking the matching family console script. Process-per-variant
preserves the JAX/XLA + GPU-memory isolation the legacy shell scripts relied on.

YAML schema
-----------
family:   one of qnet | dpg | pg | impala | apex_qnet | apex_dpg
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
"""

import argparse
import os
import signal
import subprocess
import sys

import yaml

FAMILY_SCRIPTS = {
    "qnet": "qnet",
    "dpg": "dpg",
    "pg": "pg",
    "impala": "impala",
    "apex_qnet": "apex-qnet",
    "apex_dpg": "apex-dpg",
}


def _build_args(merged):
    argv = []
    for key, value in merged.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


def _iter_commands(config, cli_overrides=None):
    cli_overrides = cli_overrides or {}
    family = config["family"]
    if family not in FAMILY_SCRIPTS:
        raise ValueError(f"unknown family '{family}', expected one of {sorted(FAMILY_SCRIPTS)}")
    script = FAMILY_SCRIPTS[family]
    base = config.get("base") or {}
    runtime = config.get("runtime") or {}
    xvfb = runtime.get("xvfb", False)
    for variant in config.get("variants") or [{}]:
        variant = variant or {}
        if not variant.get("enabled", True):
            continue
        variant_args = {k: v for k, v in variant.items() if k != "enabled"}
        command = [script, *_build_args({**base, **variant_args, **cli_overrides})]
        if xvfb:
            command = ["xvfb-run", "-a", *command]
        yield command


def main(argv=None):
    # Behave like a normal Unix tool when piped into head/grep (no BrokenPipe traceback).
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description="Run a YAML-defined experiment sweep.")
    parser.add_argument("config", help="path to a sweep YAML file")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
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
        cli_overrides[key] = value

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    runtime = config.get("runtime") or {}
    env = os.environ.copy()
    if "device" in runtime:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["device"])

    commands = list(_iter_commands(config, cli_overrides))
    failures = []
    for index, command in enumerate(commands, start=1):
        printable = " ".join(command)
        print(f"[{index}/{len(commands)}] {printable}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, env=env)
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
