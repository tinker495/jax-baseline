"""Factory helpers to create replay buffers with consistent constructor args.

This wraps constructors in `jax_baselines.common.cpprb_buffers` so callers
don't need to duplicate branching logic for prioritized / n-step / multi.
"""
from typing import Any

from jax_baselines.common.cpprb_buffers import (
    MultiPrioritizedReplayBuffer,
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)


def make_replay_buffer(
    buffer_size: int,
    observation_space: Any,
    action_shape_or_n,
    worker_size: int = 1,
    n_step: int = 1,
    gamma: float = 0.99,
    prioritized: bool = False,
    alpha: float = 0.6,
    eps: float = 1e-3,
    compress_memory: bool = False,
):
    """Create an appropriate replay buffer based on flags.

    - For prioritized + n_step -> PrioritizedNstepReplayBuffer
    - For prioritized only -> PrioritizedReplayBuffer
    - For n_step only -> NstepReplayBuffer
    - Otherwise -> ReplayBuffer
    """
    if prioritized:
        if n_step > 1:
            return PrioritizedNstepReplayBuffer(
                buffer_size,
                observation_space,
                action_shape_or_n,
                worker_size,
                n_step,
                gamma,
                alpha,
                False,
                eps,
            )
        else:
            return PrioritizedReplayBuffer(
                buffer_size, observation_space, alpha, action_shape_or_n, False, eps
            )
    else:
        if n_step > 1:
            return NstepReplayBuffer(
                buffer_size, observation_space, action_shape_or_n, worker_size, n_step, gamma
            )
        else:
            return ReplayBuffer(buffer_size, observation_space, action_shape_or_n)


def make_multi_prioritized_buffer(
    buffer_size: int,
    observation_space: Any,
    alpha: float,
    action_shape_or_n,
    n_step: int,
    gamma: float,
    manager,
    compress_memory: bool = False,
    eps: float | None = None,
):
    """Factory wrapper for MultiPrioritizedReplayBuffer used in distributed setups."""
    if eps is None:
        return MultiPrioritizedReplayBuffer(
            buffer_size,
            observation_space,
            alpha,
            action_shape_or_n,
            n_step,
            gamma,
            manager,
            compress_memory,
        )
    else:
        return MultiPrioritizedReplayBuffer(
            buffer_size,
            observation_space,
            alpha,
            action_shape_or_n,
            n_step,
            gamma,
            manager,
            compress_memory,
            eps,
        )
