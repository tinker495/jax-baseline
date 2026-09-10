from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import jax
import numpy as np


def dummy_observation(space):
    return {key: np.zeros((1, *shape), dtype=np.float32) for key, shape in space.items()}


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


class ActorCriticFeatures(TypedDict):
    actor: jax.Array
    critic: jax.Array


def observation_role_keys(
    space: Mapping[str, Sequence[int]], actor_critic: bool = False
) -> dict[str, tuple[str, ...]]:
    """Resolve canonical observation roles before embedding and network construction."""
    for key in space:
        prefix, separator, name = key.partition("_")
        if prefix not in ("unified", "actor", "critic") or not separator or not name:
            raise ValueError(f"Observation key requires unified_, actor_, or critic_: {key!r}")
    roles: tuple[Literal["actor", "critic"], ...] = (
        ("actor", "critic") if actor_critic else ("actor",)
    )
    keys = {
        role: tuple(key for key in space if key.startswith(("unified_", f"{role}_")))
        for role in roles
    }
    for role, selected in keys.items():
        if not selected:
            raise ValueError(f"Observation space has no inputs for {role}")
    return keys
