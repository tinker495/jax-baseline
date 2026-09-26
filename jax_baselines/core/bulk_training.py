"""Small helpers for chunked replay training pulses."""

from functools import lru_cache

import jax
import numpy as np

# Update-loop scans unroll this many iterations per while-loop step: a rolled scan pays a
# host sync per iteration (RLPD Walker2d: ~2.3x faster at 20 updates), while fully unrolling
# a worker-32 group (640 updates) stalled for 12+ minutes at 14 GB RSS, likely compiling.
# ponytail: one cap for every algorithm; tune per model if compile time or speed demands it.
SCAN_UNROLL = 8


def bulk_train_hook(agent):
    if not getattr(agent, "supports_bulk_training", False):
        return None
    return getattr(agent, "_train_on_bulk", None)


def update_group_iters(worker_size, train_freq):
    """Update iterations in one train pulse: a whole vector step's worth, at least one.

    Rollouts only train in whole groups (carrying the remainder), so every pulse is
    a multiple of the group and never splits into ragged chunks.
    """
    return max(1, worker_size // train_freq)


def bulk_group_size(agent):
    """Gradient updates in one whole group (gradient_steps x worker_size / train_freq)."""
    return agent.gradient_steps * update_group_iters(agent.worker_size, agent.train_freq)


def bulk_chunk_size(agent):
    """Largest chunk: max_bulk_updates_per_pulse rounded down to whole groups.

    A group is never split: when it exceeds the cap, the chunk is one whole group.
    """
    max_chunk = int(agent.max_bulk_updates_per_pulse)
    if max_chunk <= 0:
        raise ValueError("max_bulk_updates_per_pulse must be greater than 0")
    group = bulk_group_size(agent)
    return group * max(1, max_chunk // group)


def uses_bulk_pulse(agent, gradient_steps):
    if bulk_train_hook(agent) is None or gradient_steps <= 1:
        return False
    return bulk_chunk_size(agent) > 1


def bulk_chunk_schedule(agent, gradient_steps):
    """Split a pulse into chunks made of whole update groups (e.g. 20 -> (20,), not 16 + 4)."""
    group = bulk_group_size(agent)
    groups, ragged = divmod(int(gradient_steps), group)
    if ragged:
        raise ValueError(
            f"Pulse of {gradient_steps} updates is not a multiple of the update group ({group})"
        )
    chunks = bulk_chunk_plan(groups, bulk_chunk_buckets(bulk_chunk_size(agent) // group))
    # Leftover groups stay bulk chunks; single-update leftovers go to the scalar path.
    leftover = (1,) * (groups - sum(chunks)) if group > 1 else ()
    return tuple(group * chunk for chunk in chunks + leftover)


@lru_cache(maxsize=128)
def bulk_chunk_plan(gradient_steps, buckets):
    calls = [0] + [gradient_steps + 1] * gradient_steps
    scalar_counts = [0] + [gradient_steps + 1] * gradient_steps
    first_chunks = [0] * (gradient_steps + 1)

    for steps in range(1, gradient_steps + 1):
        calls[steps] = calls[steps - 1] + 1
        scalar_counts[steps] = scalar_counts[steps - 1] + 1

        for bucket in buckets:
            if bucket > steps:
                continue
            candidate_calls = calls[steps - bucket] + 1
            candidate_scalars = scalar_counts[steps - bucket]
            if (candidate_calls, candidate_scalars, -bucket) >= (
                calls[steps],
                scalar_counts[steps],
                -first_chunks[steps],
            ):
                continue
            calls[steps] = candidate_calls
            scalar_counts[steps] = candidate_scalars
            first_chunks[steps] = bucket

    chunks = []
    remaining = gradient_steps
    while remaining > 0:
        chunk_size = first_chunks[remaining]
        if chunk_size <= 0:
            remaining -= 1
            continue
        chunks.append(chunk_size)
        remaining -= chunk_size
    return tuple(chunks)


def bulk_chunk_buckets(max_chunk):
    chunk_size = int(max_chunk)
    buckets = []
    while chunk_size >= 2:
        buckets.append(chunk_size)
        chunk_size //= 2
    if buckets and buckets[-1] != 2:
        buckets.append(2)
    return tuple(buckets)


def make_train_contexts(agent, context_type, steps, chunk_size, **kwargs):
    contexts = []
    for _ in range(chunk_size):
        agent.train_steps_count += 1
        contexts.append(
            context_type(
                steps=steps,
                train_steps_count=agent.train_steps_count,
                **kwargs,
            )
        )
    return tuple(contexts)


def normalize_bulk_weight_value(value):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return value
    return value / value.max(axis=1, keepdims=True)


def reshape_bulk_value(value, chunk_size, batch_size):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return value
    if len(shape) >= 2 and shape[0] == chunk_size and shape[1] == batch_size:
        return value
    expected_flat = chunk_size * batch_size
    if shape[0] != expected_flat:
        return value
    return value.reshape((chunk_size, batch_size, *shape[1:]))


def host_priority_values(values):
    """CPU PER write-back: one explicit device->host transfer, flattened on the host."""
    return np.asarray(jax.device_get(values)).reshape(-1)


@jax.jit(static_argnames=("chunk_size", "batch_size", "flat", "obs_apply", "reward_apply"))
def _prepare_batch(
    batch, obs_stats, reward_stats, chunk_size, batch_size, flat, obs_apply, reward_apply
):
    if chunk_size is not None and "weights" in batch:
        # PER weights are max-normalized per update, as a single-update sample would be.
        weights = normalize_bulk_weight_value(
            reshape_bulk_value(batch["weights"], chunk_size, batch_size)
        )
        batch["weights"] = weights.reshape(batch["weights"].shape) if flat else weights
    if chunk_size is not None and not flat:
        batch = jax.tree.map(lambda value: reshape_bulk_value(value, chunk_size, batch_size), batch)
    if obs_apply is not None:
        batch["obses"] = obs_apply(batch["obses"], obs_stats)
        batch["nxtobses"] = obs_apply(batch["nxtobses"], obs_stats)
    if reward_apply is not None:
        batch["rewards"] = reward_apply(batch["rewards"], reward_stats)
    return batch


def prepare_replay_batch(
    data, *, chunk_size=None, batch_size=None, flat=False, obs_rms=None, rewards=None
):
    """Replay sample -> learner batch with one explicit transfer and one compiled call.

    Bulk samples (``chunk_size`` updates of ``batch_size``) get PER weights normalized per
    update and become ``(chunk, batch, ...)``, or stay flat for learners that slice the
    chunk themselves (``flat``). Observations and rewards are normalized with the current
    device statistics whatever the replay storage. ``indexes`` stays where the replay
    produced it for the priority write-back.
    """
    batch = jax.device_put({name: value for name, value in data.items() if name != "indexes"})
    batch = _prepare_batch(
        batch,
        None if obs_rms is None else obs_rms.stats,
        None if rewards is None else rewards.stats,
        chunk_size=chunk_size,
        batch_size=batch_size,
        flat=flat,
        obs_apply=None if obs_rms is None else obs_rms.apply,
        reward_apply=None if rewards is None else rewards.apply,
    )
    if "indexes" in data:
        batch["indexes"] = data["indexes"]
    return batch
