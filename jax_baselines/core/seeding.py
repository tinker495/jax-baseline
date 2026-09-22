"""Core PRNG and process seeding helpers.

Environment-object seeding lives in the repo-local ``env_builder`` adapter because
it depends on concrete Gymnasium-style spaces.
"""

import os
import random
from typing import Optional

import jax
import numpy as np


@jax.jit
def _split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Keep unpacking inside the compiled call to avoid separate slice dispatches.
    keys = jax.random.split(key)
    return keys[0], keys[1]


def key_gen(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = _split_key(key)
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
