"""Factory helpers to create replay buffers with consistent constructor args.

This wraps constructors in `replay_memory.cpprb_buffers` so callers
don't need to duplicate branching logic for prioritized / n-step / multi.
"""

from typing import Any

from replay_memory.cpprb_buffers import (
    MultiPrioritizedReplayBuffer,
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from replay_memory.frame_buffers import (
    FrameStackReplayBuffer,
    PrioritizedFrameStackReplayBuffer,
)


def _frame_compress_applicable(observation_space, worker_size, n_step, n_frames):
    """Frame-level n-step compaction needs a single frame-stacked image modality,
    a single worker (contiguous transition stream), and a real n-step horizon.
    cpprb's stack_compress cannot compact the n-step next_obs, so this is the only
    way to get the full ~8x saving for n-step image replay."""
    return (
        worker_size == 1
        and n_step > 1
        and len(observation_space) == 1
        and len(observation_space[0]) >= 3
        and observation_space[0][-1] % n_frames == 0
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
    n_frames: int = 4,
):
    """Create an appropriate replay buffer based on flags.

    - compress_memory + n_step + single-worker image -> Frame(Prioritized)StackReplayBuffer
      (stores one frame per observation; reconstructs the stack and the n-step
      next_obs by index, so 1e6 Atari n-step replay costs ~7GB instead of ~35GB)
    - For prioritized + n_step -> PrioritizedNstepReplayBuffer
    - For prioritized only -> PrioritizedReplayBuffer
    - For n_step only -> NstepReplayBuffer
    - Otherwise -> ReplayBuffer
    """
    if compress_memory and _frame_compress_applicable(
        observation_space, worker_size, n_step, n_frames
    ):
        if prioritized:
            return PrioritizedFrameStackReplayBuffer(
                buffer_size,
                observation_space,
                action_shape_or_n,
                n_step,
                gamma,
                alpha,
                eps,
                n_frames,
            )
        return FrameStackReplayBuffer(
            buffer_size, observation_space, action_shape_or_n, n_step, gamma, n_frames
        )

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
                compress_memory,
                eps,
            )
        else:
            return PrioritizedReplayBuffer(
                buffer_size,
                observation_space,
                alpha,
                action_shape_or_n,
                compress_memory,
                eps,
            )
    else:
        if n_step > 1:
            return NstepReplayBuffer(
                buffer_size,
                observation_space,
                action_shape_or_n,
                worker_size,
                n_step,
                gamma,
                compress_memory,
            )
        else:
            return ReplayBuffer(buffer_size, observation_space, action_shape_or_n, compress_memory)


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


def make_worker_replay_buffer(local_size: int, *, env_dict: dict, n_s: dict | None = None):
    """Create the APE-X worker-local replay buffer from shared buffer metadata."""
    return ReplayBuffer(local_size, env_dict=env_dict, n_s=n_s)
