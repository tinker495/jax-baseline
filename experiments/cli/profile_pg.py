"""Profile real vectorized PG rollouts, excluding compilation and trace overhead."""

import argparse
import gzip
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from functools import wraps
from importlib.metadata import version
from pathlib import Path

import yaml

from experiments.cli._common import load_runtime_env
from experiments.cli.exp import _build_args


def summarize_trace(trace_path: Path, expected_steps: int) -> dict:
    trace_file = max(trace_path.glob("plugins/profile/*/*.trace.json.gz"))
    with gzip.open(trace_file, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    processes = {
        event["pid"]: event["args"]["name"]
        for event in events
        if "ph" in event and event["ph"] == "M" and event["name"] == "process_name"
    }
    gpu_pids = {pid for pid, name in processes.items() if "/device:GPU:" in name}
    complete = [event for event in events if "ph" in event and event["ph"] == "X"]
    host_counts = Counter(event["name"] for event in complete if event["pid"] not in gpu_pids)
    transfers = {}
    for name in ("MemcpyH2D", "MemcpyD2H", "MemcpyD2D"):
        copies = [event for event in complete if event["pid"] in gpu_pids and event["name"] == name]
        sizes = []
        for event in copies:
            match = re.search(r"\bsize:(\d+)\b", event["args"]["memcpy_details"])
            if match is None:
                raise ValueError(f"GPU transfer lacks byte size: {event}")
            sizes.append(int(match[1]))
        transfers[name] = {
            "gpu_event_count": len(copies),
            "bytes": sum(sizes),
            "gpu_duration_ms_sum": sum(event["dur"] for event in copies) / 1000,
            "host_duration_ms_sum": sum(
                event["dur"]
                for event in complete
                if event["pid"] not in gpu_pids and event["name"] == name
            )
            / 1000,
            "small_at_most_256_bytes_count": sum(size <= 256 for size in sizes),
        }
    return {
        "trace": str(trace_file),
        "trace_event_count": len(events),
        "gpu_processes": [processes[pid] for pid in gpu_pids],
        "expected_rollout_steps": expected_steps,
        "observed_environment_steps": host_counts["environment.step"],
        "observed_training_updates": host_counts["training.update"],
        "complete_rollout": bool(gpu_pids)
        and host_counts["environment.step"] == expected_steps
        and host_counts["training.update"] == 1,
        "transfers": transfers,
        "timing_note": "Counts and bytes use GPU events only; matching host events are not "
        "double counted. Summed GPU durations are not wall time. Compare only complete rollouts.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/pg_mjlab_go1.yaml")
    )
    parser.add_argument("--algo", choices=("PPO", "SPO"), default="PPO")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/profile_pg"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--env-jax-arrays", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-device-rollout", action="store_true")
    parser.add_argument("--memory-backend", choices=("auto", "cpu", "gpu"))
    options = parser.parse_args(argv)
    if options.warmup < 1 or options.rollouts < 1:
        parser.error("--warmup and --rollouts must be positive")
    load_runtime_env()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    config = yaml.safe_load(options.config.read_text())
    if config["family"] != "pg":
        parser.error("--config must select the pg family")
    variants = [variant for variant in config["variants"] if variant["algo"] == options.algo]
    if len(variants) != 1:
        parser.error("--algo must match exactly one YAML variant")
    merged = {
        **config["base"],
        **{k: v for k, v in variants[0].items() if k != "enabled"},
    }
    if options.env_jax_arrays is not None:
        merged["env_jax_arrays"] = options.env_jax_arrays
    if options.memory_backend is not None:
        merged["memory_backend"] = options.memory_backend
    if "runtime" in config and "device" in config["runtime"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["device"])

    import jax
    import torch

    from experiments.checkpoint_store import FileCheckpointStore
    from experiments.cli._run import _close_agent_envs, resolve_maker
    from experiments.cli.pg import PG_RUNNER
    from jax_baselines.core.runtime_adapters import NoOpLoggerRun
    from jax_baselines.core.training_session import RunContext

    if not all(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU profiling requires a JAX GPU device")
    pg_parser = argparse.ArgumentParser()
    PG_RUNNER.add_args(pg_parser)
    args = pg_parser.parse_args(_build_args(merged))
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "source.diff").write_text(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], text=True)
    )
    untracked_sources = [
        Path(path)
        for path in subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "jax_baselines",
                "env_builder",
                "model_builder",
                "experiments",
                "replay_memory",
            ],
            text=True,
        ).splitlines()
        if Path(path).suffix == ".py"
    ]
    for path in untracked_sources:
        snapshot = options.output / "source" / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    spec = PG_RUNNER.algos[args.algo]
    env_builder, policy_kwargs = PG_RUNNER.build_env(args)
    agent = spec.resolve_cls(args)(
        env_builder,
        resolve_maker(PG_RUNNER, spec, args),
        policy_kwargs=policy_kwargs,
        checkpoint_store=FileCheckpointStore(),
        **spec.build(args),
    )
    agent.prepare_run(int(args.steps))
    samples = []
    losses = []
    host_seconds = defaultdict(float)
    call_counts = defaultdict(int)
    device_checks = defaultdict(int)
    rollout_index = 0
    measuring = True

    def synchronize():
        jax.block_until_ready((agent.params, agent.opt_state))
        torch.cuda.synchronize()

    def timed(name, function):
        @wraps(function)
        def call(*call_args, **kwargs):
            if options.require_device_rollout and name == "buffer.add":
                require_device("buffer.add", (call_args, kwargs))
                if agent.obs_rms is not None:
                    require_device(
                        "observation.statistics",
                        (agent.obs_rms.means, agent.obs_rms.vars, agent.obs_rms.count),
                    )
            start = time.perf_counter()
            with jax.profiler.TraceAnnotation(name):
                result = function(*call_args, **kwargs)
            if measuring and rollout_index >= options.warmup:
                host_seconds[name] += time.perf_counter() - start
                call_counts[name] += 1
            if options.require_device_rollout and name == "buffer.get_buffer":
                require_device("buffer.get_buffer", result)
            return result

        return call

    def require_device(label, tree):
        for leaf in jax.tree.leaves(tree):
            if not isinstance(leaf, jax.Array) or any(
                device.platform != "gpu" for device in leaf.devices()
            ):
                raise RuntimeError(f"{label} requires GPU JAX arrays, got {type(leaf).__name__}")
            device_checks[label] += 1

    agent.actions = timed("policy.actions", agent.actions)
    agent.normalize_observation = timed("observation.normalize", agent.normalize_observation)
    if agent.obs_rms is not None:
        agent.obs_rms.update = timed("observation.update_statistics", agent.obs_rms.update)
    agent.env.step = timed("environment.step", agent.env.step)
    agent.env.get_result = timed("environment.get_result", agent.env.get_result)
    agent.env.current_obs = timed("environment.current_obs", agent.env.current_obs)
    agent.buffer.add = timed("buffer.add", agent.buffer.add)
    agent.buffer.get_buffer = timed("buffer.get_buffer", agent.buffer.get_buffer)
    train_step = agent.train_step
    synchronize()
    rollout_start = time.perf_counter()

    def train(steps, logger_run=None):
        nonlocal rollout_index, rollout_start
        start = time.perf_counter()
        with jax.profiler.TraceAnnotation("training.update"):
            result = train_step(steps, logger_run=logger_run)
            synchronize()
        end = time.perf_counter()
        if not math.isfinite(float(result)):
            raise RuntimeError(f"non-finite loss at rollout {rollout_index}: {result}")
        if measuring:
            elapsed = end - rollout_start
            if rollout_index >= options.warmup:
                samples.append(elapsed)
                losses.append(float(result))
                host_seconds["training.update_synchronized"] += end - start
                call_counts["training.update_synchronized"] += 1
            rollout_index += 1
            print(
                f"rollout={rollout_index} seconds={elapsed:.6f} loss={float(result):.6f}",
                flush=True,
            )
        rollout_start = time.perf_counter()
        return result

    agent.train_step = train
    rollout_steps = agent.worker_size * agent.batch_size
    logger = NoOpLoggerRun(str(options.output))
    try:
        agent.learn_VectorizedEnv(
            RunContext(
                logger,
                eval_freq=10**18,
                pbar=range(
                    rollout_steps,
                    (options.warmup + options.rollouts + 1) * rollout_steps,
                    agent.worker_size,
                ),
                log_interval=10**18,
            )
        )
        if len(samples) != options.rollouts:
            raise RuntimeError(f"expected {options.rollouts} measured rollouts, got {len(samples)}")
        summary = {
            "config": str(options.config),
            "arguments": vars(args),
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "diff": str(options.output / "source.diff"),
            "untracked_source_snapshots": [
                str(options.output / "source" / path) for path in untracked_sources
            ],
            "versions": {name: version(name) for name in ("jax", "jaxlib", "torch", "mjlab")},
            "devices": [str(device) for device in jax.devices()],
            "gpu": torch.cuda.get_device_name(),
            "warmup_rollouts": options.warmup,
            "measured_rollouts": options.rollouts,
            "transitions_per_rollout": rollout_steps,
            "rollout_seconds": samples,
            "median_rollout_seconds": statistics.median(samples),
            "transitions_per_second": rollout_steps * len(samples) / sum(samples),
            "losses": losses,
            "component_host_seconds": dict(host_seconds),
            "component_calls": dict(call_counts),
            "require_device_rollout": options.require_device_rollout,
            "memory_backend": agent.memory_backend,
            "device_array_leaves_checked": dict(device_checks),
            "timing_note": "Component host times overlap and exclude pending GPU work, except "
            "training.update_synchronized. Rollout times synchronize JAX and Torch at updates. "
            "Warmup, evaluation, checkpoints, logging I/O and the trace pass are excluded.",
        }
        if options.compare is not None:
            previous = json.loads(options.compare.read_text())
            previous_args = {
                "env_jax_arrays": False,
                "memory_backend": "auto",
                **previous["arguments"],
            }
            current_args = {"env_jax_arrays": False, "memory_backend": "auto", **vars(args)}
            if previous_args.keys() != current_args.keys():
                raise ValueError("comparison requires the same PG argument schema")
            changed = {
                key: {"before": previous_args[key], "after": current_args[key]}
                for key in previous_args
                if previous_args[key] != current_args[key]
            }
            if changed.keys() - {"env_jax_arrays", "logdir", "memory_backend"}:
                raise ValueError(f"comparison requires matching PG arguments: {changed}")
            summary["comparison"] = {
                "baseline": str(options.compare),
                "changed_arguments": changed,
                "throughput_ratio": summary["transitions_per_second"]
                / previous["transitions_per_second"],
            }
        (options.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if options.trace:
            measuring = False
            trace_path = options.output / "trace"
            profiler_options = jax.profiler.ProfileOptions()
            profiler_options.python_tracer_level = 0
            with jax.profiler.trace(
                str(trace_path), create_perfetto_trace=True, profiler_options=profiler_options
            ):
                agent.learn_VectorizedEnv(
                    RunContext(
                        logger,
                        eval_freq=10**18,
                        pbar=range(rollout_steps, 2 * rollout_steps, agent.worker_size),
                        log_interval=10**18,
                    )
                )
                synchronize()
            summary["trace"] = str(trace_path)
            (options.output / "trace_summary.json").write_text(
                json.dumps(summarize_trace(trace_path, agent.batch_size), indent=2) + "\n"
            )
            summary["trace_summary"] = str(options.output / "trace_summary.json")
            (options.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"trace={trace_path}", flush=True)
    finally:
        _close_agent_envs(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
