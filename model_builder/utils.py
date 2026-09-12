from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import numpy as np

from model_builder.model_config import (
    DEFAULT_MLP,
    EMBEDDING_MODES,
    MLPConfig,
    ModelConfig,
    resolve_model_config,
)


def qnet_model_kwargs(
    policy_kwargs: dict | None,
    *,
    default: MLPConfig = DEFAULT_MLP,
    allowed_embeddings: tuple[str, ...] = EMBEDDING_MODES,
) -> dict:
    options = {} if policy_kwargs is None else dict(policy_kwargs)
    if {"node", "hidden_n", "embedding_mode"} & options.keys():
        raise ValueError("Use model JSON to configure the Q-network layers and embedding_mode")
    options["network"] = resolve_model_config(
        options.pop("model", None), default, allowed_embeddings=allowed_embeddings
    )
    return options


def dummy_observation(space):
    return {key: np.zeros((1, *shape), dtype=np.float32) for key, shape in space.items()}


def get_critic_apply_fn(critic_apply, shared_preproc_apply):
    def apply_fn(critic_params, actor_params, key, observations, *args):
        shared_features = jax.lax.stop_gradient(
            shared_preproc_apply(actor_params, key, observations)
        )
        return critic_apply(critic_params, key, observations, shared_features, *args)

    return apply_fn


def print_flax_model_summary(enabled, key, *models):
    if not enabled:
        return

    for model, *inputs in models:
        print(model.tabulate(key, *inputs))


def print_haiku_model_summary(enabled, *models):
    if not enabled:
        return

    import haiku as hk

    for model, *inputs in models:
        print(hk.experimental.tabulate(model)(*inputs))


def observation_role_keys(
    space: Mapping[str, Sequence[int]], role: Literal["actor", "critic"] = "actor"
) -> tuple[str, ...]:
    """Resolve canonical observation roles before embedding and network construction."""
    if role not in ("actor", "critic"):
        raise ValueError(f"Unknown observation role: {role!r}")
    for key in space:
        prefix, separator, name = key.partition("_")
        if prefix not in ("unified", "actor", "critic") or not separator or not name:
            raise ValueError(f"Observation key requires unified_, actor_, or critic_: {key!r}")
    keys = tuple(key for key in space if key.startswith(("unified_", f"{role}_")))
    if not keys:
        raise ValueError(f"Observation space has no inputs for {role}")
    return keys


def split_actor_critic_kwargs(
    policy_kwargs: dict | None,
    *,
    actor_default: ModelConfig = DEFAULT_MLP,
    critic_default: ModelConfig = DEFAULT_MLP,
    allowed_types: tuple[str, ...] | None = None,
    allowed_embeddings: tuple[str, ...] = EMBEDDING_MODES,
) -> tuple[dict, dict]:
    """Resolve independent actor/critic descriptions at the builder boundary."""
    options = {} if policy_kwargs is None else dict(policy_kwargs)
    if set(options) & {"node", "actor_node", "critic_node", "hidden_n", "embedding_mode"}:
        raise ValueError("Use actor_model and critic_model JSON descriptions for network settings")
    actor = resolve_model_config(
        options.pop("actor_model", None),
        actor_default,
        allowed_types=allowed_types,
        allowed_embeddings=allowed_embeddings,
    )
    critic = resolve_model_config(
        options.pop("critic_model", None),
        critic_default,
        allowed_types=allowed_types,
        allowed_embeddings=allowed_embeddings,
    )
    return {**options, "network": actor}, {**options, "network": critic}
