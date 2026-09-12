"""GPU replay sampling and learner dispatch through a functional adapter seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from jax_baselines.core.bulk_training import (
    flatten_bulk_batch,
    normalize_bulk_weights,
    reshape_bulk_batch,
)
from jax_baselines.core.normalization import (
    FlashSACRewardNormalizer,
    RewardNormalizer,
    _normalize_device_observations,
)


class ReplayTrainingBuffer(Protocol):
    def train(
        self,
        batch: ReplayTrainingBatch,
        learner: Callable[..., Any],
        args: tuple,
        *,
        bulk: bool,
        priority_index: int | None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ReplayTrainingBatch:
    """A pending sample; normalization statistics are snapshots for this pulse."""

    replay: ReplayTrainingBuffer
    sample_size: int
    beta: float
    chunk_size: int = 0
    observation_statistics: tuple[dict[str, jax.Array], dict[str, jax.Array]] | None = None
    reward_statistics: tuple[jax.Array, jax.Array | None] | None = None


def reward_normalization_statistics(normalizer: RewardNormalizer | None):
    if normalizer is None:
        return None
    minimum_scale = (
        normalizer.max_abs_return / normalizer.normalized_G_max
        if isinstance(normalizer, FlashSACRewardNormalizer)
        else None
    )
    return normalizer.rms.vars["return"], minimum_scale


def train_replay_batch(learner, data, *args, priority_index=-1):
    if isinstance(data, ReplayTrainingBatch):
        return data.replay.train(data, learner, args, bulk=False, priority_index=priority_index)
    return learner(*args, **data)


def train_replay_bulk(learner, data, carry, keys, steps, *, priority_index=-1):
    if isinstance(data, ReplayTrainingBatch):
        return data.replay.train(
            data,
            learner,
            (carry, keys, steps),
            bulk=True,
            priority_index=priority_index,
        )
    return learner(carry, keys, steps, data)


def train_replay(
    state,
    key,
    args,
    observation_statistics,
    reward_statistics,
    *,
    sample,
    update,
    learner,
    sample_size,
    beta,
    chunk_size,
    bulk,
    priority_index,
):
    key, data = sample(state, key, sample_size, beta)
    chunk_size = chunk_size or (1 if bulk else 0)
    if chunk_size:
        data = normalize_bulk_weights(
            reshape_bulk_batch(data, chunk_size, sample_size // chunk_size)
        )
    if observation_statistics is not None:
        means, variances = observation_statistics
        data["obses"] = _normalize_device_observations(data["obses"], means, variances)
        data["nxtobses"] = _normalize_device_observations(data["nxtobses"], means, variances)
    if reward_statistics is not None:
        variance, minimum_scale = reward_statistics
        scale = jnp.sqrt(variance + 1e-8)
        if minimum_scale is not None:
            scale = jnp.maximum(scale, minimum_scale)
        data["rewards"] = data["rewards"] / scale
    if bulk:
        result = learner(*args, data)
    else:
        if chunk_size:
            data = flatten_bulk_batch(data)
        result = learner(*args, **data)
    if update is not None:
        if priority_index is None:
            raise ValueError("Prioritized replay requires a learner priority output")
        priorities = result[1][priority_index] if bulk else result[priority_index]
        indexes = jnp.asarray(data["indexes"], dtype=jnp.int32).reshape(-1)
        priorities = jnp.asarray(priorities, dtype=jnp.float32).reshape(-1)
        if indexes.shape != priorities.shape or not indexes.size:
            raise ValueError("Priority indexes and values must have matching non-empty shapes")
        state = update(state, indexes, priorities)
    return state, key, result
