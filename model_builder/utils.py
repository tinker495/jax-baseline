from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import numpy as np


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
    policy_kwargs: dict | None, *, actor_node: int = 256, critic_node: int = 256
) -> tuple[dict, dict]:
    """Resolve independent network widths at the model-builder boundary."""
    options = {} if policy_kwargs is None else dict(policy_kwargs)
    if "node" in options:
        raise ValueError("Use actor_node and critic_node to configure actor-critic networks")
    actor_node = options.pop("actor_node", actor_node)
    critic_node = options.pop("critic_node", critic_node)
    for name, value in (("actor_node", actor_node), ("critic_node", critic_node)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {**options, "node": actor_node}, {**options, "node": critic_node}
