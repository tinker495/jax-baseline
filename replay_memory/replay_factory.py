"""Factory helpers to create replay buffers with consistent constructor args.

This wraps constructors in `replay_memory.cpprb_buffers` so callers
don't need to duplicate branching logic for prioritized / n-step / multi.
"""

from jax_baselines.core.replay_protocol import (
    LocalReplayNeed,
    SelfPredictionReplayNeed,
    SharedPrioritizedReplayNeed,
)
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
from replay_memory.impala_buffers import EpochBuffer
from replay_memory.transition_buffers import (
    PrioritizedTransitionReplayBuffer,
    TransitionReplayBuffer,
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


def _validate_self_prediction_replay_need(need: SelfPredictionReplayNeed) -> None:
    unsupported = []
    if need.compress_observations:
        unsupported.append("compress_observations=True")
    if need.worker_size != 1:
        unsupported.append(f"worker_size={need.worker_size}")
    if not unsupported:
        return
    joined = ", ".join(unsupported)
    raise ValueError(
        "SelfPredictionReplayNeed does not support "
        f"{joined}; transition replay buffers are single-worker and uncompressed"
    )


def make_replay_buffer(need: LocalReplayNeed):
    """Create an appropriate replay buffer based on flags.

    - compress_memory + n_step + single-worker image -> Frame(Prioritized)StackReplayBuffer
      (stores one frame per observation; reconstructs the stack and the n-step
      next_obs by index, so 1e6 Atari n-step replay costs ~7GB instead of ~35GB)
    - For prioritized + n_step -> PrioritizedNstepReplayBuffer
    - For prioritized only -> PrioritizedReplayBuffer
    - For n_step only -> NstepReplayBuffer
    - Otherwise -> ReplayBuffer
    """
    buffer_size = need.buffer_size
    observation_space = need.observation_space
    action_shape_or_n = need.action_shape_or_n
    worker_size = need.worker_size
    n_step = need.n_step
    gamma = need.gamma
    prioritized = need.priority is not None
    alpha = need.priority.alpha if need.priority is not None else 0.6
    eps = need.priority.eps if need.priority is not None else 1e-3
    compress_memory = need.compress_observations
    n_frames = need.n_frames

    if isinstance(need, SelfPredictionReplayNeed):
        _validate_self_prediction_replay_need(need)
        if prioritized:
            return PrioritizedTransitionReplayBuffer(
                buffer_size,
                observation_space,
                action_shape_or_n,
                prediction_depth=need.prediction_depth,
                alpha=alpha,
                eps=eps,
            )
        return TransitionReplayBuffer(
            buffer_size,
            observation_space,
            action_shape_or_n,
            prediction_depth=need.prediction_depth,
        )

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


def make_multi_prioritized_buffer(need: SharedPrioritizedReplayNeed):
    """Factory wrapper for MultiPrioritizedReplayBuffer used in distributed setups."""
    return MultiPrioritizedReplayBuffer(
        need.buffer_size,
        need.observation_space,
        need.priority.alpha,
        need.action_shape_or_n,
        need.n_step,
        need.gamma,
        need.manager,
        need.compress_observations,
        eps=need.priority.eps,
    )


def make_worker_replay_buffer(local_size: int, *, env_dict: dict, n_s: dict | None = None):
    """Create the APE-X worker-local replay buffer from shared buffer metadata."""
    return ReplayBuffer(local_size, env_dict=env_dict, n_s=n_s)


def make_impala_worker_buffer(local_size: int, *, env_dict: dict, n_s: dict | None = None):
    """Create the IMPALA worker-local rollout buffer (stores V-trace ``log_prob``).

    Satisfies the ``WorkerReplayBufferFactory`` seam used by the IMPALA worker.
    ``n_s`` is accepted for protocol uniformity but unused: the IMPALA rollout is
    a flat single-step sequence, not an n-step buffer.
    """
    return EpochBuffer(local_size, env_dict)
