"""Device-resident replay with Flashbax storage, sampling and priority trees."""

import dataclasses
from math import prod

import flashbax
import jax
import jax.numpy as jnp
import numpy as np
from flashbax.buffers import prioritised_trajectory_buffer, sum_tree, trajectory_buffer

from jax_baselines.core.replay_protocol import LocalReplayNeed, SelfPredictionReplayNeed


def _resum_paths(tree, leaf_indexes, valid):
    """Rewrite the ancestors of written leaves as exact sums of their children.

    Flashbax propagates priority *deltas* up the sum tree in float32, so internal sums drift
    from their leaves. A root above the true total lets sampling land on empty
    (zero-priority) slots, whose importance weight is infinite. Recomputing every touched
    path after each write keeps all internal nodes equal to the sum of their children.
    """
    nodes = tree.nodes
    first_leaf = 2**tree.tree_depth - 1
    node = jnp.where(valid, leaf_indexes, 0) + first_leaf
    for _ in range(tree.tree_depth):
        node = (node - 1) // 2
        nodes = nodes.at[node].set(nodes[2 * node + 1] + nodes[2 * node + 2])
    return tree.replace(nodes=nodes)


def _stratified_indexes(tree, key, batch_size):
    """Flashbax's stratified sum-tree sampling (same query values), never reaching empty leaves.

    Float rounding can leave a query at or past a node's left sum when the right subtree is
    empty; descending left there keeps every sample on a positive-priority leaf.
    """
    query_keys = jax.random.split(key, batch_size)
    bounds = jnp.linspace(0.0, 1.0, batch_size + 1)
    query = jax.vmap(jax.random.uniform, in_axes=(0, None, None, 0, 0))(
        query_keys, (), jnp.float32, bounds[:-1], bounds[1:]
    )
    nodes = tree.nodes
    query = query * nodes[0]
    node = jnp.zeros(batch_size, dtype=jnp.int32)
    for _ in range(tree.tree_depth):
        left = 2 * node + 1
        left_sum = nodes[left]
        go_left = (query < left_sum) | (nodes[left + 1] <= 0)
        query = jnp.where(go_left, query, query - left_sum)
        node = jnp.where(go_left, left, left + 1)
    return node - (2**tree.tree_depth - 1)


class FlashbaxReplayBuffer:
    def __init__(self, need: LocalReplayNeed):
        if need.buffer_size < 1 or need.worker_size < 1 or need.n_step < 1:
            raise ValueError("buffer_size, worker_size and n_step must be positive")
        if not isinstance(need.observation_space, dict) or not need.observation_space:
            raise ValueError("observation_space must be a non-empty dict")
        if not 0 <= need.gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if need.priority is not None and (
            not 0 <= need.priority.alpha <= 1 or need.priority.eps < 0
        ):
            raise ValueError("priority alpha must be in [0, 1] and eps must be nonnegative")
        if need.device is None:
            try:
                self.device = jax.devices("gpu")[0]
            except (RuntimeError, IndexError) as error:
                raise RuntimeError("GPU replay requires an available JAX GPU device") from error
        else:
            self.device = need.device
        self.max_size = need.buffer_size
        self.worker_size = need.worker_size
        self.prediction_depth = (
            need.prediction_depth if isinstance(need, SelfPredictionReplayNeed) else None
        )
        self.n_step = need.n_step if self.prediction_depth is None else 1
        self.gamma = need.gamma
        self.priority = need.priority
        self.observation_space = need.observation_space
        self.action_shape = (
            (need.action_shape_or_n,)
            if isinstance(need.action_shape_or_n, int)
            else tuple(need.action_shape_or_n)
        )
        self.seed = need.seed
        if self.priority is None:
            self.buffer = flashbax.make_item_buffer(
                max_length=self.max_size, min_length=1, sample_batch_size=1, add_batches=True
            )
        else:
            self.buffer = flashbax.make_prioritised_trajectory_buffer(
                add_batch_size=1,
                sample_batch_size=1,
                sample_sequence_length=1,
                period=1,
                min_length_time_axis=1,
                max_length_time_axis=self.max_size,
                priority_exponent=self.priority.alpha,
                device=self.device.platform,
            )
        observation_specs = {
            key: jax.ShapeDtypeStruct(
                (self.worker_size, *shape), jnp.uint8 if len(shape) >= 3 else jnp.float32
            )
            for key, shape in self.observation_space.items()
        }
        self._input_specs = (
            {
                "obses": observation_specs,
                "actions": jax.ShapeDtypeStruct(
                    (self.worker_size, *self.action_shape), jnp.float32
                ),
                "rewards": jax.ShapeDtypeStruct((self.worker_size, 1), jnp.float32),
                "nxtobses": observation_specs,
                "terminateds": jax.ShapeDtypeStruct((self.worker_size, 1), jnp.float32),
            },
            jax.ShapeDtypeStruct((self.worker_size,), bool),
            jax.ShapeDtypeStruct((self.worker_size,), bool),
        )
        self._add_compiled = jax.jit(self._add, donate_argnums=(0, 1))
        self._sample_compiled = jax.jit(self._sample, static_argnums=(2, 3))
        self._update_compiled = jax.jit(self._update_priorities, donate_argnums=(0,))
        self.clear()

    def clear(self):
        with jax.default_device(self.device):
            example = {
                "obses": {
                    key: jnp.zeros(shape, dtype=jnp.uint8 if len(shape) >= 3 else jnp.float32)
                    for key, shape in self.observation_space.items()
                },
                "actions": jnp.zeros(self.action_shape, dtype=jnp.float32),
                "rewards": jnp.zeros((1,), dtype=jnp.float32),
                "nxtobses": {
                    key: jnp.zeros(shape, dtype=jnp.uint8 if len(shape) >= 3 else jnp.float32)
                    for key, shape in self.observation_space.items()
                },
                "terminateds": jnp.zeros((1,), dtype=jnp.float32),
            }
            if self.prediction_depth is not None:
                example["sequence_index"] = jnp.zeros((), dtype=jnp.int32)
                example["episode_ends"] = jnp.zeros((), dtype=bool)
            self.state = jax.jit(self.buffer.init)(example)
            self._pending = (
                jax.tree.map(
                    lambda value: jnp.zeros(
                        (self.worker_size, self.n_step, *value.shape), dtype=value.dtype
                    ),
                    {key: example[key] for key in ("obses", "actions", "rewards")},
                ),
                jnp.zeros(self.worker_size, dtype=jnp.int32),
            )
            self._key = jax.random.PRNGKey(self.seed)
            self._all_active = jnp.ones(self.worker_size, dtype=bool)
            self._not_truncated = jnp.zeros(self.worker_size, dtype=bool)
        self._sample_ready = False

    def __len__(self):
        """Explicit host query; add/sample never poll the device's write pointer."""
        is_full, current_index = jax.device_get((self.state.is_full, self.state.current_index))
        return self.max_size if is_full else int(current_index)

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False, store_mask=None):
        if (
            not isinstance(obs_t, dict)
            or not isinstance(nxtobs_t, dict)
            or obs_t.keys() != self.observation_space.keys()
            or nxtobs_t.keys() != obs_t.keys()
        ):
            raise ValueError("Observation keys must match observation_space")

        batch = {
            "obses": dict(obs_t),
            "actions": action,
            "rewards": reward,
            "nxtobses": dict(nxtobs_t),
            "terminateds": terminated,
        }
        mask = self._all_active if store_mask is None else store_mask
        truncations = self._not_truncated if truncated is False else truncated
        inputs = (batch, truncations, mask)
        for value, spec in zip(jax.tree.leaves(inputs), jax.tree.leaves(self._input_specs)):
            if prod(np.shape(value)) != prod(spec.shape):
                raise ValueError(
                    f"Replay input shape {np.shape(value)} does not match {spec.shape}"
                )
        # Placement happens at this boundary (a no-op for inputs already on the device);
        # dtype casts and reshapes run inside the compiled add.
        self.state, self._pending = self._add_compiled(
            self.state, self._pending, *jax.device_put(inputs, self.device)
        )

    def _add(self, state, pending, batch, truncated, active):
        batch, truncated, active = jax.tree.map(
            lambda value, spec: jnp.asarray(value, spec.dtype).reshape(spec.shape),
            (batch, truncated, active),
            self._input_specs,
        )
        if self.prediction_depth is not None:
            batch = {
                **batch,
                "sequence_index": state.current_index.reshape((1,)),
                "episode_ends": batch["terminateds"].reshape(-1).astype(bool) | truncated,
            }
        if self.n_step == 1:
            return self._append(state, batch, active), pending
        staged, lengths = pending
        workers = jnp.arange(self.worker_size)
        staged = jax.tree.map(
            lambda history, value: history.at[workers, lengths].set(value),
            staged,
            {key: batch[key] for key in ("obses", "actions", "rewards")},
        )
        lengths = lengths + active
        boundary = (batch["terminateds"].reshape(-1).astype(bool) | truncated) & active
        positions = jnp.arange(self.n_step)
        ready = (
            active[:, None]
            & (positions < lengths[:, None])
            & (boundary[:, None] | ((lengths == self.n_step)[:, None] & (positions == 0)))
        )
        distances = positions[None, :] - positions[:, None]
        discounts = jnp.where(distances >= 0, self.gamma ** jnp.maximum(distances, 0), 0)
        rewards = jnp.einsum(
            "st,wtk->wsk",
            discounts,
            staged["rewards"] * (positions < lengths[:, None])[..., None],
        )
        items = {
            **staged,
            "rewards": rewards,
            "nxtobses": jax.tree.map(
                lambda value: jnp.broadcast_to(
                    value[:, None], (self.worker_size, self.n_step, *value.shape[1:])
                ),
                batch["nxtobses"],
            ),
            "terminateds": jnp.broadcast_to(
                batch["terminateds"][:, None], (self.worker_size, self.n_step, 1)
            ),
        }
        state = self._append(
            state,
            jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), items),
            ready.reshape(-1),
        )
        shift = (lengths == self.n_step) & active & ~boundary
        staged = jax.tree.map(
            lambda value: jnp.where(
                shift.reshape((self.worker_size,) + (1,) * (value.ndim - 1)),
                jnp.roll(value, -1, axis=1),
                value,
            ),
            staged,
        )
        return state, (staged, jnp.where(boundary, 0, lengths - shift))

    def _append(self, state, batch, valid):
        # Flashbax's native add has a static length. Compact masked workers and
        # flushed episode tails without copying a variable item count to the CPU.
        ranks = jnp.cumsum(valid, dtype=jnp.int32) - 1
        count = jnp.sum(valid, dtype=jnp.int32)
        keep = valid & (ranks >= count - self.max_size)
        indexes = jnp.where(keep, (state.current_index + ranks) % self.max_size, self.max_size)
        experience = jax.tree.map(
            lambda storage, values: storage.at[0, indexes].set(values, mode="drop"),
            state.experience,
            batch,
        )
        if self.priority is not None:
            tree = state.sum_tree_state
            tree = sum_tree.set_batch_bincount(
                tree,
                jnp.where(keep, indexes, tree.nodes.size + 1),
                jnp.where(keep, tree.max_recorded_priority, 0),
            )
            state = state.replace(
                sum_tree_state=_resum_paths(tree, indexes, keep),
                running_index=state.running_index + count,
            )
        return state.replace(
            experience=experience,
            current_index=(state.current_index + count) % self.max_size,
            is_full=state.is_full | (state.current_index + count >= self.max_size),
        )

    def sample(self, batch_size: int, beta=0.4):
        if batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError("batch_size must be positive and beta must be in [0, 1]")
        if not self._sample_ready:
            if len(self) == 0:
                raise ValueError("Cannot sample from empty replay")
            self._sample_ready = True
        self._key, batch = self._sample_compiled(self.state, self._key, batch_size, beta)
        return batch

    def _sample(self, state, key, batch_size, beta):
        key, sample_key = jax.random.split(key)
        if self.priority is None:
            sample = trajectory_buffer.sample(state, sample_key, batch_size, 1, 1)
            batch = jax.tree.map(lambda value: value[:, 0], sample.experience)
            if self.prediction_depth is not None:
                batch = self._sample_sequence(state, batch)
            return key, batch
        # One add batch and period 1: a sum-tree leaf index is the storage time index.
        indexes = _stratified_indexes(state.sum_tree_state, sample_key, batch_size)
        batch = jax.tree.map(lambda value: value[0, indexes], state.experience)
        sampled_priorities = sum_tree.get_batch(state.sum_tree_state, indexes)
        if self.prediction_depth is not None:
            batch = self._sample_sequence(state, batch)
            minimum = jnp.min(sampled_priorities)
        else:
            # ponytail: O(capacity) minimum preserves cpprb's global IS normalization;
            # use a min tree if prioritized replay profiling makes this significant.
            leaves = state.sum_tree_state.nodes[2**state.sum_tree_state.tree_depth - 1 :][
                : self.max_size
            ]
            minimum = jnp.min(jnp.where(leaves > 0, leaves, jnp.inf))
        return key, {
            **batch,
            "weights": (minimum / sampled_priorities) ** beta,
            "indexes": indexes,
        }

    def _sample_sequence(self, state, starts):
        assert self.prediction_depth is not None
        offsets = jnp.arange(self.prediction_depth)
        indexes = (starts["sequence_index"][:, None] + offsets) % self.max_size
        episode_ends = state.experience["episode_ends"][0, indexes]
        available = jnp.where(
            state.is_full,
            (state.current_index - starts["sequence_index"] - 1) % self.max_size + 1,
            state.current_index - starts["sequence_index"],
        )
        filled = (offsets < available[:, None]) & (
            jnp.cumsum(episode_ends, axis=1) - episode_ends == 0
        )
        # Repeat the final valid transition for padding; never expose a reset or
        # overwritten row as a future observation, including truncation bootstrap.
        indexes = (
            starts["sequence_index"][:, None]
            + jnp.minimum(offsets, jnp.sum(filled, axis=1)[:, None] - 1)
        ) % self.max_size
        batch = jax.tree.map(lambda value: value[0, indexes], state.experience)
        return {
            "obses": jax.tree.map(
                lambda obs, next_obs: jnp.concatenate((obs[:, :1], next_obs), axis=1),
                batch["obses"],
                batch["nxtobses"],
            ),
            "actions": batch["actions"],
            "rewards": batch["rewards"][..., 0],
            "terminateds": batch["terminateds"][..., 0].astype(bool),
            "filled": filled,
        }

    def update_priorities(self, indexes, priorities):
        if self.priority is None:
            raise ValueError("Priority updates require prioritized replay")
        if prod(np.shape(indexes)) != prod(np.shape(priorities)) or not prod(np.shape(indexes)):
            raise ValueError("Priority indexes and values must have matching non-empty shapes")
        self.state = self._update_compiled(
            self.state, *jax.device_put((indexes, priorities), self.device)
        )

    def _update_priorities(self, state, indexes, priorities):
        assert self.priority is not None
        indexes = indexes.reshape(-1).astype(jnp.int32)
        state = prioritised_trajectory_buffer.set_priorities(
            state,
            indexes,
            jnp.abs(priorities.reshape(-1).astype(jnp.float32)) + self.priority.eps,
            self.priority.alpha,
            self.device.platform,
        )
        return dataclasses.replace(
            state,
            sum_tree_state=_resum_paths(
                state.sum_tree_state, indexes, jnp.ones(indexes.shape, dtype=bool)
            ),
        )
