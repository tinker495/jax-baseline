from typing import Optional

import gymnasium as gym

from jax_baselines.core.seeding import key_gen, seed_prngs, set_global_seeds

__all__ = ["key_gen", "seed_env", "seed_prngs", "set_global_seeds"]


def seed_env(env: gym.Env, seed: Optional[int]) -> None:
    """Seed a gymnasium environment and its action/observation spaces."""
    if seed is None:
        return

    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "seed"):
        space.seed(seed)

    space = getattr(env, "observation_space", None)
    if space is not None and hasattr(space, "seed"):
        space.seed(seed)
