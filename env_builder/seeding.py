import os
import random
from typing import Optional

import gymnasium as gym
import jax
import numpy as np


def key_gen(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def seed_prngs(seed: Optional[int]) -> None:
    """Seed Python's random and NumPy RNGs if a seed is provided."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed global RNGs (Python, NumPy, hash) for reproducibility."""
    if seed is None:
        return

    seed_prngs(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def seed_env(env: gym.Env, seed: Optional[int]) -> None:
    """Seed a gymnasium environment and its action/observation spaces."""
    if seed is None:
        return

    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    except Exception:
        pass

    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "seed"):
        try:
            space.seed(seed)
        except Exception:
            pass

    space = getattr(env, "observation_space", None)
    if space is not None and hasattr(space, "seed"):
        try:
            space.seed(seed)
        except Exception:
            pass
