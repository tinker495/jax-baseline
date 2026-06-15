"""Repo-local environment adapter package.

Concrete Gymnasium/EnvPool/Atari environment construction lives here.
The Algorithm Core consumes only the core-facing env contracts in
``jax_baselines.common.env_protocols`` and legacy compatibility shims.
"""

from env_builder.env_builder import (
    EnvPoolVectorizedEnv,
    GymVectorizedEnv,
    get_env_builder,
)
from env_builder.seeding import key_gen, seed_env, seed_prngs, set_global_seeds
from jax_baselines.common.env_protocols import Env, SingleEnv, VectorizedEnv

__all__ = [
    "Env",
    "SingleEnv",
    "VectorizedEnv",
    "EnvPoolVectorizedEnv",
    "GymVectorizedEnv",
    "get_env_builder",
    "key_gen",
    "seed_env",
    "seed_prngs",
    "set_global_seeds",
]
