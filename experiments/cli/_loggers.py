"""``--logger`` backend selection for the experiment CLIs.

Maps the ``--logger`` name to a logger factory (the callable shape
``(run_name, experiment_name, local_dir, agent) -> logger``).

Concrete backends live under ``experiments.loggers``; their third-party
dependencies (``wandb``, ``aim``) are optional dependencies (in the dev group) imported lazily, so the
default (``tensorboard``) requires neither to be installed. Selecting a backend
whose extra is not installed raises a clear, actionable error.

This is the single shared CLI helper: :func:`add_logger_args` is called once by
the family runners, so the flags are not duplicated across the six family
parsers.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace
from functools import partial

from experiments.cli._common import LOGGER_ENV
from model_builder.model_config import MLPConfig, ResidualConfig, model_config_dict

DEFAULT_LOGGER = "tensorboard"
LOGGER_BACKENDS = ("tensorboard", "wandb", "aim")


def add_logger_args(parser: ArgumentParser) -> None:
    """Add the shared ``--logger`` selector plus each backend's config flags.

    Defaults read the environment (populated from ``.env`` by
    :func:`experiments.cli._common.load_runtime_env`), so the backend and its
    config can be set in a file; an explicit CLI flag always overrides the env.
    """
    parser.add_argument(
        "--logger",
        type=str,
        default=os.environ.get(LOGGER_ENV, DEFAULT_LOGGER),
        choices=list(LOGGER_BACKENDS),
        help=f"logging backend (default tensorboard; env ${LOGGER_ENV})",
    )
    # Weights & Biases (--logger wandb); standard WANDB_* env vars are the defaults.
    # The W&B project is `experiment_name` (--experiment_name), the cross-backend
    # experiment grouping. WANDB_API_KEY is read by the wandb SDK once loaded from .env.
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity / team (env $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE"),
        choices=["online", "offline", "disabled"],
        help="W&B mode (env $WANDB_MODE; default: W&B's own default)",
    )
    # Aim (--logger aim).
    parser.add_argument(
        "--aim_repo",
        type=str,
        default=os.environ.get("AIM_REPO"),
        help="Aim repo path (env $AIM_REPO; default: ./.aim)",
    )


def resolve_logger_factory(args: Namespace, *, policy_kwargs: dict[str, object] | None = None):
    """Return the ``LoggerFactory`` selected by ``args.logger`` (lazy import)."""
    model_names: list[str] = []
    model_hparams: dict[str, str] = {}
    options = {} if policy_kwargs is None else policy_kwargs
    for role in ("model", "actor_model", "critic_model"):
        if role not in options:
            continue
        config = options[role]
        if not isinstance(config, (MLPConfig, ResidualConfig)):
            raise TypeError(f"{role} must be a validated model configuration")
        description = model_config_dict(config)
        model_names.append(f"{role.removesuffix('_model').title()}-{description['type']}")
        model_hparams[f"{role}_type"] = description["type"]
        model_hparams[role] = json.dumps(description, sort_keys=True)

    name = args.logger
    if name == "tensorboard":
        from experiments.runtime_adapters import TensorboardLogger

        factory = (
            partial(TensorboardLogger, extra_hparams=model_hparams)
            if model_hparams
            else TensorboardLogger
        )
    elif name == "wandb":
        from experiments.loggers.wandb_logger import make_wandb_logger_factory

        factory = make_wandb_logger_factory(args, extra_hparams=model_hparams)
    elif name == "aim":
        from experiments.loggers.aim_logger import make_aim_logger_factory

        factory = make_aim_logger_factory(args, extra_hparams=model_hparams)
    else:
        raise SystemExit(f"unknown --logger '{name}', expected one of {list(LOGGER_BACKENDS)}")
    if not model_names:
        return factory

    def model_logger(run_name, experiment_name, local_dir, agent):
        return factory(f"{'_'.join(model_names)}_{run_name}", experiment_name, local_dir, agent)

    return model_logger
