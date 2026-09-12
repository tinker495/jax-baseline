"""Backend-independent reproduction records for experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from argparse import Namespace
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import jax

from jax_baselines.core.env_protocols import EnvInfo
from jax_baselines.core.training_session import eval_freq_from_count
from model_builder.model_config import MLPConfig, ResidualConfig, model_config_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CONTEXT_ENV = "JAXBL_SWEEP_CONTEXT"


def collect_run_metadata(
    args: Namespace,
    *,
    command: list[str],
    policy_kwargs: dict[str, object],
    training_envs: list[EnvInfo],
    evaluation_env: EnvInfo | None,
    shared_evaluation_env: bool,
    algorithm_parameters: dict,
) -> dict[str, object]:
    models = {}
    for role in ("model", "actor_model", "critic_model"):
        if role not in policy_kwargs:
            continue
        config = policy_kwargs[role]
        if not isinstance(config, (MLPConfig, ResidualConfig)):
            raise TypeError(f"{role} must be a validated model configuration")
        models[role] = model_config_dict(config)
    if not models:
        raise ValueError("The model builder did not record its resolved model configuration")

    git: dict[str, object]
    try:
        repository = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        git = {"available": False, "reason": "git executable is not installed"}
    else:
        if repository.returncode != 0:
            git = {"available": False, "reason": repository.stderr.strip()}
        else:
            root = Path(repository.stdout.strip())
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
            diff = subprocess.check_output(
                ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"],
                cwd=root,
                text=True,
            )
            untracked = (
                subprocess.check_output(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
                )
                .decode()
                .split("\0")[:-1]
            )
            for name in untracked:
                addition = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--binary",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-index",
                        "--",
                        "/dev/null",
                        name,
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if addition.returncode not in (0, 1):
                    raise RuntimeError(
                        f"Cannot preserve untracked source {name}: {addition.stderr}"
                    )
                diff += addition.stdout
            git = {
                "available": True,
                "commit": commit.strip(),
                "dirty": bool(diff),
                "diff": diff,
                "untracked_files": untracked,
            }

    lock = PROJECT_ROOT / "uv.lock"
    lock_text = lock.read_text(encoding="utf-8") if lock.is_file() else None
    distributed = evaluation_env is None
    runner = command[0]
    evaluation = (
        {"mode": "rollout_only", "episodes": None, "seed_rule": None}
        if distributed
        else {
            "mode": "frozen_policy",
            "episodes": args.eval_eps,
            "requested_evaluation_count": args.eval_num,
            "interval_step_slots": eval_freq_from_count(
                args.eval_num, int(args.steps), training_envs[0]["worker_num"]
            ),
            "final_evaluation": True,
            "shared_training_environment": shared_evaluation_env,
            "seed_rule": "training state restored after evaluation"
            if shared_evaluation_env
            else "train seed + 1 initially; subsequent resets advance the same RNG stream",
            "policy": "epsilon=0; parameter noise remains enabled when configured"
            if runner == "qnet"
            else "argmax or Gaussian mean"
            if runner == "pg"
            else "evaluation actor without exploration noise",
        }
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "working_directory": str(Path.cwd()),
        "arguments": vars(args).copy(),
        "algorithm_parameters": algorithm_parameters,
        "models": models,
        "git": git,
        "dependencies": {
            "python": platform.python_version(),
            "packages": dict(
                sorted((dist.metadata["Name"], dist.version) for dist in distributions())
            ),
            "lockfile": lock_text,
            "lockfile_sha256": hashlib.sha256(lock_text.encode()).hexdigest()
            if lock_text
            else None,
        },
        "runtime": {
            "platform": platform.platform(),
            "devices": [
                {"platform": device.platform, "kind": device.device_kind, "id": device.id}
                for device in jax.devices()
            ],
            "environment_variables": {
                name: os.environ[name]
                for name in (
                    "JAX_PLATFORMS",
                    "JAX_ENABLE_X64",
                    "JAX_DEFAULT_PRNG_IMPL",
                    "XLA_FLAGS",
                    "CUDA_VISIBLE_DEVICES",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "MUJOCO_GL",
                )
                if name in os.environ
            },
            "jax_enable_x64": jax.config.values["jax_enable_x64"],
            "jax_default_prng_impl": jax.config.values["jax_default_prng_impl"],
        },
        "environments": {"training": training_envs, "evaluation": evaluation_env},
        "evaluation": evaluation,
        "measurements": {
            "requested_steps": int(args.steps),
            "steps_unit": "learner_iterations" if distributed else "environment_step_slots",
            "progress/env_steps": "training transitions, excluding evaluation and autoreset dummy rows",
            "progress/update_steps": (
                "performed learner minibatch update rounds; "
                "actor/critic updates in one round are counted once"
            ),
            "time/elapsed_seconds": "monotonic seconds since training started, including in-training evaluation",
            "episode_reward": "episode return of rewards received by the algorithm",
            "original_reward": "unclipped full-game return when supplied by the environment adapter",
        },
        "sweep": json.loads(os.environ[SWEEP_CONTEXT_ENV])
        if SWEEP_CONTEXT_ENV in os.environ
        else None,
    }


def tracking_experiment_name(experiment_name: str, metadata: dict[str, object] | None) -> str:
    if metadata is None:
        return experiment_name
    sweep = metadata["sweep"]
    if sweep is None:
        return experiment_name
    if not isinstance(sweep, dict):
        raise TypeError("run metadata sweep must be a mapping")
    category = dict(sweep)["category"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("run metadata sweep category must be a nonempty string")
    return category


def write_run_metadata(local_dir: str, metadata: dict[str, object] | None) -> None:
    if metadata is None:
        return
    path = Path(local_dir) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Run metadata: {path.resolve()}", flush=True)
