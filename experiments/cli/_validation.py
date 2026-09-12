"""Shared, side-effect-free argument checks for direct runs and sweep preflight."""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace
from importlib.util import find_spec
from typing import TYPE_CHECKING

from experiments.cli._loggers import add_logger_args
from experiments.optimizers import select_optimizer
from model_builder.model_config import (
    DEFAULT_MLP,
    MLPConfig,
    load_model_config,
    resolve_model_config,
)

if TYPE_CHECKING:
    from experiments.cli._run import DistributedFamilyRunner, FamilyRunner


def runner_parser(
    runner: FamilyRunner | DistributedFamilyRunner, *, prog: str | None = None
) -> ArgumentParser:
    parser = ArgumentParser(prog=prog, allow_abbrev=False)
    runner.add_args(parser)
    add_logger_args(parser)
    return parser


def parse_runner_args(
    runner: FamilyRunner | DistributedFamilyRunner,
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
) -> Namespace:
    parser = runner_parser(runner, prog=prog)
    args = parser.parse_args(argv)
    if args.algo not in runner.algos:
        parser.error(f"unknown algo {args.algo!r}, expected one of {sorted(runner.algos)}")
    if args.model_lib not in ("flax", "haiku"):
        parser.error("--model_lib must be flax or haiku")
    module = (
        f"{runner.maker_pkg.format(lib=args.model_lib)}.{runner.algos[args.algo].builder}_builder"
    )
    if find_spec(module) is None:
        parser.error(f"unsupported combo: algo={args.algo} model_lib={args.model_lib}")

    values = vars(args)
    for name, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"--{name} must be finite")
    for name in {
        "steps",
        "worker",
        "batch",
        "batch_size",
        "batch_num",
        "n_step",
        "mini_batch",
        "train_freq",
        "gradient_steps",
        "max_bulk_updates_per_pulse",
        "update_freq",
        "sample_size",
        "epoch_num",
        "n_support",
        "critic_num",
        "actor_update_period",
        "eval_eps",
        "target_update",
        "env_episode_length",
    } & values.keys():
        value = values[name]
        if value is not None and (value < 1 or int(value) != value):
            parser.error(f"--{name} must be a positive integer")
    for name in {"seed", "learning_starts", "eval_num", "buffer_size"} & values.keys():
        value = values[name]
        if value < 0 or int(value) != value:
            parser.error(f"--{name} must be a nonnegative integer")
    for name in {"learning_rate", "optimizer_eps", "max_grad_norm"} & values.keys():
        if values[name] is not None and values[name] <= 0:
            parser.error(f"--{name} must be positive")
    for name in {
        "gamma",
        "lamda",
        "target_update_tau",
        "final_eps",
        "initial_eps",
    } & values.keys():
        if not 0 <= values[name] <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if not args.env.strip():
        parser.error("--env must be nonempty")
    if "env_backend" in values:
        if args.env_jax_arrays and args.env_backend != "mjlab":
            parser.error("--env_jax_arrays requires --env_backend mjlab")
        if args.env_reuse_for_eval and args.env_backend == "envpool":
            parser.error("--env_reuse_for_eval does not support EnvPool")

    try:
        select_optimizer(args.optimizer, args.learning_rate)
        for name in {"model", "actor_model", "critic_model"} & values.keys():
            if values[name] is not None:
                builder = runner.algos[args.algo].builder
                allowed_types = ("mlp",)
                if args.model_lib == "flax" and runner.maker_pkg.endswith(".dpg"):
                    if builder == "flashsac":
                        allowed_types = ("flashsac",)
                    elif builder != "xqc":
                        allowed_types = ("mlp", "simba", "simbav2")
                model = resolve_model_config(
                    load_model_config(values[name]),
                    DEFAULT_MLP,
                    allowed_types=allowed_types,
                    allowed_embeddings=("normal",)
                    if args.model_lib == "haiku" or builder == "flashsac"
                    else ("normal", "resnet"),
                )
                if builder == "td7" and isinstance(model, MLPConfig) and not model.layers:
                    raise ValueError("TD7 models require at least one hidden layer")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.steps = int(args.steps)
    if "buffer_size" in values:
        args.buffer_size = int(args.buffer_size)
    return args
