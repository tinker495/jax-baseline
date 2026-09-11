"""Device-resident replay with Flashbax storage, sampling and priority trees."""

from math import prod

import flashbax
import jax
import jax.numpy as jnp
from flashbax.buffers import prioritised_trajectory_buffer, sum_tree, trajectory_buffer

from jax_baselines.core.replay_protocol import LocalReplayNeed, SelfPredictionReplayNeed


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
        return int(jnp.where(self.state.is_full, self.max_size, self.state.current_index))

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False, store_mask=None):
        if (
            not isinstance(obs_t, dict)
            or not isinstance(nxtobs_t, dict)
            or obs_t.keys() != self.observation_space.keys()
            or nxtobs_t.keys() != obs_t.keys()
        ):
            raise ValueError("Observation keys must match observation_space")

        # Placement happens at this boundary; JAX inputs already on this device stay there.
        def array(value, shape, dtype):
            result = (
                value
                if isinstance(value, jax.Array)
                and value.dtype == dtype
                and value.devices() == {self.device}
                else jnp.asarray(value, dtype=dtype, device=self.device)
            )
            if result.size != prod(shape):
                raise ValueError(f"Replay input shape {result.shape} does not match {shape}")
            return result if result.shape == shape else result.reshape(shape)

        batch = {
            "obses": {
                key: array(
                    obs_t[key],
                    (self.worker_size, *shape),
                    jnp.uint8 if len(shape) >= 3 else jnp.float32,
                )
                for key, shape in self.observation_space.items()
            },
            "actions": array(action, (self.worker_size, *self.action_shape), jnp.float32),
            "rewards": array(reward, (self.worker_size, 1), jnp.float32),
            "nxtobses": {
                key: array(
                    nxtobs_t[key],
                    (self.worker_size, *shape),
                    jnp.uint8 if len(shape) >= 3 else jnp.float32,
                )
                for key, shape in self.observation_space.items()
            },
            "terminateds": array(terminated, (self.worker_size, 1), jnp.float32),
        }
        mask = (
            self._all_active if store_mask is None else array(store_mask, (self.worker_size,), bool)
        )
        truncations = (
            self._not_truncated
            if truncated is False
            else array(truncated, (self.worker_size,), bool)
        )
        self.state, self._pending = self._add_compiled(
            self.state, self._pending, batch, truncations, mask
        )

    def _add(self, state, pending, batch, truncated, active):
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
            state = state.replace(
                sum_tree_state=sum_tree.set_batch_bincount(
                    tree,
                    jnp.where(keep, indexes, tree.nodes.size + 1),
                    jnp.where(keep, tree.max_recorded_priority, 0),
                ),
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
        sample = prioritised_trajectory_buffer.prioritised_sample(
            state, sample_key, batch_size, 1, 1
        )
        batch = jax.tree.map(lambda value: value[:, 0], sample.experience)
        sampled_priorities = sum_tree.get_batch(state.sum_tree_state, sample.indices)
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
            "indexes": sample.indices,
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
        indexes = jnp.asarray(indexes, dtype=jnp.int32, device=self.device).reshape(-1)
        priorities = jnp.asarray(priorities, dtype=jnp.float32, device=self.device).reshape(-1)
        if indexes.shape != priorities.shape or not indexes.size:
            raise ValueError("Priority indexes and values must have matching non-empty shapes")
        self.state = self._update_compiled(self.state, indexes, priorities)

    def _update_priorities(self, state, indexes, priorities):
        assert self.priority is not None
        return prioritised_trajectory_buffer.set_priorities(
            state,
            indexes,
            jnp.abs(priorities) + self.priority.eps,
            self.priority.alpha,
            self.device.platform,
        )
