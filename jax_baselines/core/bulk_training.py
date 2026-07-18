"""Small helpers for chunked replay training pulses."""

from functools import lru_cache

import jax
import numpy as np


def bulk_train_hook(agent):
    if not getattr(agent, "supports_bulk_training", False):
        return None
    return getattr(agent, "_train_on_bulk", None)


def bulk_chunk_size(agent):
    max_chunk = int(agent.max_bulk_updates_per_pulse)
    if max_chunk <= 0:
        raise ValueError("max_bulk_updates_per_pulse must be greater than 0")
    return max_chunk


def uses_bulk_pulse(agent, gradient_steps):
    if bulk_train_hook(agent) is None or gradient_steps <= 1:
        return False
    max_chunk = bulk_chunk_size(agent)
    return max_chunk > 1


def bulk_chunk_schedule(agent, gradient_steps):
    max_chunk = bulk_chunk_size(agent)
    return tuple(iter_bulk_chunk_sizes(gradient_steps, max_chunk))


def iter_bulk_chunk_sizes(gradient_steps, max_chunk):
    """Yield bounded bulk chunk sizes, avoiding scalar leftovers when supported."""
    buckets = bulk_chunk_buckets(max_chunk)
    yield from bulk_chunk_plan(int(gradient_steps), buckets)


@lru_cache(maxsize=128)
def bulk_chunk_plan(gradient_steps, buckets):
    calls = [0] + [gradient_steps + 1] * gradient_steps
    scalar_counts = [0] + [gradient_steps + 1] * gradient_steps
    first_chunks = [0] * (gradient_steps + 1)

    for steps in range(1, gradient_steps + 1):
        calls[steps] = calls[steps - 1] + 1
        scalar_counts[steps] = scalar_counts[steps - 1] + 1

        for bucket in buckets:
            if bucket <= steps:
                candidate_calls = calls[steps - bucket] + 1
                candidate_scalars = scalar_counts[steps - bucket]
                if _is_better_chunk_plan(
                    candidate_calls,
                    candidate_scalars,
                    bucket,
                    calls[steps],
                    scalar_counts[steps],
                    first_chunks[steps],
                ):
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


def _is_better_chunk_plan(
    candidate_calls,
    candidate_scalars,
    candidate_chunk,
    current_calls,
    current_scalars,
    current_chunk,
):
    return (
        candidate_calls < current_calls
        or (candidate_calls == current_calls and candidate_scalars < current_scalars)
        or (
            candidate_calls == current_calls
            and candidate_scalars == current_scalars
            and candidate_chunk > current_chunk
        )
    )


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


def reshape_bulk_batch(data, chunk_size, batch_size):
    return jax.tree.map(lambda value: reshape_bulk_value(value, chunk_size, batch_size), data)


def normalize_bulk_weights(data):
    weights = data.get("weights")
    if weights is None:
        return data
    data = dict(data)
    data["weights"] = normalize_bulk_weight_value(weights)
    return data


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


def iter_bulk_batches(data, contexts):
    """Yield per-mini-update slices from a bulk replay sample."""
    for index in range(len(contexts)):
        yield jax.tree.map(lambda value: slice_bulk_value(value, index), data)


def slice_bulk_value(value, index):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return value
    return value[index]


def flatten_bulk_batch(data):
    """Collapse ``(chunk, batch, ...)`` bulk data into ``(chunk * batch, ...)``."""
    return jax.tree.map(flatten_bulk_value, data)


def flatten_bulk_value(value):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return value
    return value.reshape((shape[0] * shape[1], *shape[2:]))


def flatten_priority_values(values):
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) <= 1:
        return values
    return values.reshape((-1,))


def host_priority_values(values):
    return np.asarray(jax.device_get(flatten_priority_values(values)))
