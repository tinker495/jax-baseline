"""Frame-level transition replay buffer for image / frame-stacked observations.

cpprb's ``stack_compress`` cannot compact the n-step ``next_obs`` (its rows are n
steps apart, so the sliding-window reconstruction breaks — verified empirically),
and it only shares frames within one contiguous stream, so interleaved vector-env
rows get no compression at all. This buffer instead stores a single newest frame
per observation and reconstructs both the frame-stack and the n-step next
observation by index, so an n-step Atari replay at 1e6 costs ~7GB instead of
~35GB (full ``next_obs``).

Design (dopamine ``OutOfGraphReplayBuffer`` style):
  * each worker owns a contiguous ring of ``size // worker_size`` slots, so a frame
    stack or n-step window never crosses workers; flat slot = worker * capacity + ring.
  * store one newest frame per transition; reconstruct an S-frame stack by gathering
    S consecutive frames of the same frame stream, padding at the stream start (matching
    gym ``FrameStack``, which repeats the reset frame and keeps the newest frame in the
    last channel). A stream restarts only when the env resets the stack: EnvPool's
    episodic-life boundaries keep it, so a row whose obs equals the previous boundary's
    next_obs continues the stream (``stack_step``).
  * compute the n-step reward / next index / done at sample time from per-step
    reward+terminated, bounded by terminated/truncated.

Scope: single image modality. The replay factory falls back to cpprb for vector or
multi-modal observations.
"""

import random

import numpy as np


def _frame_geometry(observation_space, n_frames):
    shapes = list(observation_space.values())
    if len(shapes) != 1 or len(shapes[0]) != 3:
        raise ValueError(
            "FrameStackReplayBuffer supports a single image modality "
            f"(H, W, C*stack); got observation_space={observation_space}"
        )
    h, w, stacked_c = shapes[0]
    if stacked_c % n_frames != 0:
        raise ValueError(f"stacked channels {stacked_c} not divisible by n_frames {n_frames}")
    return int(h), int(w), stacked_c // n_frames


class FrameStackReplayBuffer:
    def __init__(
        self,
        size: int,
        observation_space: dict,
        action_space=1,
        n_step: int = 1,
        gamma: float = 0.99,
        n_frames: int = 4,
        worker_size: int = 1,
    ):
        self.worker_size = int(worker_size)
        self.capacity = int(size) // self.worker_size
        self.max_size = self.capacity * self.worker_size
        self.n_step = int(n_step)
        self.gamma = gamma
        self.n_frames = int(n_frames)
        self.observation_key = next(iter(observation_space))
        self.h, self.w, self.cf = _frame_geometry(observation_space, self.n_frames)
        self.stacked_c = self.cf * self.n_frames
        self.action_shape = (
            (action_space,) if isinstance(action_space, int) else tuple(action_space)
        )

        self._frame = np.zeros((self.max_size, self.h, self.w, self.cf), dtype=np.uint8)
        self._action = np.zeros((self.max_size, *self.action_shape), dtype=np.float32)
        self._reward = np.zeros((self.max_size,), dtype=np.float32)
        self._terminated = np.zeros((self.max_size,), dtype=np.bool_)
        self._truncated = np.zeros((self.max_size,), dtype=np.bool_)
        self._stack_step = np.zeros((self.max_size,), dtype=np.int32)
        self._boundary_next = {}

        # Per-worker transitions ever added (monotonic), the next row's stack step if its
        # stream continues, and the full next_obs of a just-stored boundary row.
        self._count = np.zeros(self.worker_size, dtype=np.int64)
        self._next_stack_step = np.zeros(self.worker_size, dtype=np.int32)
        self._after_boundary = np.zeros(self.worker_size, dtype=np.bool_)
        self._boundary_obs = np.zeros(
            (self.worker_size, self.h, self.w, self.stacked_c), dtype=np.uint8
        )
        self._discounts = (gamma ** np.arange(self.n_step)).astype(np.float64)

    # ---- bookkeeping ----------------------------------------------------
    def __len__(self) -> int:
        return int(np.minimum(self._count, self.capacity).sum())

    def _slot(self, workers, abs_idx):
        return workers * self.capacity + abs_idx % self.capacity

    def _oldest(self):
        return np.maximum(0, self._count - self.capacity)

    def _sample_bounds(self):
        """Per-worker [start, stop) of absolute indices whose stack and n-step window are intact.

        Once a ring wraps, its oldest n_frames - 1 rows lost the frames their stacks need;
        the newest n_step rows still wait for their next frame.
        """
        oldest = self._oldest()
        start = oldest + np.where(oldest > 0, self.n_frames - 1, 0)
        return start, np.maximum(start, self._count - self.n_step)

    def episode_end(self):
        # add() already advances the episode on terminated/truncated; kept for
        # interface parity with the cpprb buffers.
        pass

    # ---- add ------------------------------------------------------------
    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False, store_mask=None):
        """Store one row per worker; ``store_mask`` drops vector autoreset dummy rows."""
        workers = (
            np.arange(self.worker_size)
            if store_mask is None
            else np.flatnonzero(np.asarray(store_mask, dtype=bool))
        )
        if not len(workers):
            return
        terminated = np.reshape(np.asarray(terminated, dtype=bool), self.worker_size)[workers]
        truncated = np.reshape(np.asarray(truncated, dtype=bool), self.worker_size)[workers]
        boundary = terminated | truncated
        obs = np.asarray(obs_t[self.observation_key])[workers]
        new_stream = self._count[workers] == 0
        after = np.flatnonzero(self._after_boundary[workers])
        new_stream[after] |= ~np.all(
            obs[after] == self._boundary_obs[workers[after]], axis=(1, 2, 3)
        )
        if new_stream.any():
            fresh = obs[new_stream]
            if not np.array_equal(fresh, np.tile(fresh[..., -self.cf :], self.n_frames)):
                raise ValueError(
                    "frame stack neither continues the previous row nor repeats its reset "
                    "frame, so FrameStackReplayBuffer cannot reconstruct it"
                )
        stack_step = np.where(new_stream, 0, self._next_stack_step[workers])
        slots = self._slot(workers, self._count[workers])
        for slot in slots:
            self._boundary_next.pop(int(slot), None)
        self._frame[slots] = obs[..., -self.cf :]
        self._action[slots] = np.reshape(
            np.asarray(action, dtype=np.float32), (self.worker_size, *self.action_shape)
        )[workers]
        self._reward[slots] = np.reshape(np.asarray(reward), self.worker_size)[workers]
        self._terminated[slots] = terminated
        self._truncated[slots] = truncated
        self._stack_step[slots] = stack_step
        if boundary.any():
            next_obs = np.asarray(nxtobs_t[self.observation_key])[workers[boundary]]
            self._boundary_obs[workers[boundary]] = next_obs
            for slot, frame in zip(slots[boundary], next_obs[..., -self.cf :], strict=True):
                self._boundary_next[int(slot)] = frame.copy()
        self._count[workers] += 1
        self._next_stack_step[workers] = stack_step + 1
        self._after_boundary[workers] = boundary
        return workers

    # ---- reconstruction -------------------------------------------------
    def _gather_stack(self, workers, abs_idx, lo):
        """Reconstruct the S-frame stacked observation for per-worker absolute indices.

        workers, abs_idx, lo: int arrays of shape (B,). Frames are stacked oldest-first
        along the last axis (newest in the final channel block), padded at the
        stream start by repeating its first frame.
        """
        e = self._stack_step[self._slot(workers, abs_idx)]  # stream frames before this obs
        back = np.arange(self.n_frames - 1, -1, -1)  # oldest first
        off = np.minimum(np.minimum(back, e[:, None]), (abs_idx - lo)[:, None])
        # One (B, S) gather, then interleave S into channels: strided per-frame writes
        # into the channel axis were ~6x slower.
        frames = self._frame[self._slot(workers[:, None], abs_idx[:, None] - off)]
        return np.moveaxis(frames, 1, 3).reshape(len(abs_idx), self.h, self.w, self.stacked_c)

    def _nstep(self, workers, a):
        """Vectorised n-step over per-worker absolute start indices a (B,).

        Returns (reward (B,), next_idx (B,), done (B,), boundary (B,))."""
        b = a.shape[0]
        reward = np.zeros(b, dtype=np.float64)
        steps = np.zeros(b, dtype=np.int64)
        done = np.zeros(b, dtype=np.float32)
        active = np.ones(b, dtype=bool)
        for k in range(self.n_step):
            idx = self._slot(workers, a + k)
            reward += active * self._discounts[k] * self._reward[idx]
            steps += active
            term = self._terminated[idx] & active
            trunc = self._truncated[idx] & active
            done = np.where(term, 1.0, done)
            active = active & ~(term | trunc)
        hit_boundary = ~active
        next_idx = a + steps - hit_boundary.astype(np.int64)
        return reward.astype(np.float32), next_idx, done, hit_boundary

    def _gather(self, workers, a):
        lo = self._oldest()[workers]
        reward, next_idx, done, hit_boundary = self._nstep(workers, a)
        obses = {self.observation_key: self._gather_stack(workers, a, lo)}
        next_stack = self._gather_stack(workers, next_idx, lo)
        for row in np.flatnonzero(hit_boundary):
            next_stack[row, ..., : -self.cf] = next_stack[row, ..., self.cf :]
            next_stack[row, ..., -self.cf :] = self._boundary_next[
                int(self._slot(workers[row], next_idx[row]))
            ]
        nxtobses = {self.observation_key: next_stack}
        return {
            "obses": obses,
            "actions": self._action[self._slot(workers, a)],
            "rewards": reward[:, None],
            "nxtobses": nxtobses,
            "terminateds": done[:, None],
        }

    def _sample_indices(self, batch_size):
        start, stop = self._sample_bounds()
        ready = stop - start
        offsets = np.cumsum(ready) - ready
        flat = np.random.randint(0, ready.sum(), size=batch_size)
        workers = np.searchsorted(offsets + ready, flat, side="right")
        return workers, start[workers] + flat - offsets[workers]

    def sample(self, batch_size: int):
        start, stop = self._sample_bounds()
        if (stop - start).sum() <= 0:
            raise ValueError("Cannot sample: no fully-observed n-step transitions yet")
        return self._gather(*self._sample_indices(batch_size))


class _SumTree:
    """Proportional-priority sum tree indexed directly by ring slot."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.max_priority = 1.0

    def set(self, leaf, p):
        idx = leaf + self.capacity - 1
        self.tree[idx] = p
        idx = (idx - 1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2 * idx + 1] + self.tree[2 * idx + 2]
            if idx == 0:
                break
            idx = (idx - 1) // 2
        self.max_priority = max(self.max_priority, p)

    def total(self):
        return self.tree[0]

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1)


class PrioritizedFrameStackReplayBuffer(FrameStackReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: dict,
        action_space=1,
        n_step: int = 1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        eps: float = 1e-4,
        n_frames: int = 4,
        worker_size: int = 1,
    ):
        super().__init__(
            size, observation_space, action_space, n_step, gamma, n_frames, worker_size
        )
        self.alpha = alpha
        self.eps = eps
        self._tree = _SumTree(self.max_size)

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False, store_mask=None):
        workers = super().add(
            obs_t, action, reward, nxtobs_t, terminated, truncated, store_mask=store_mask
        )
        if workers is None:
            return
        start, stop = self._sample_bounds()
        for w in workers:
            # The just-added row is not sampleable yet (its n-step window is unobserved),
            # nor are wrapped rows whose stacks lost frames; a row becomes sampleable
            # n_step adds later and gets max priority then (deferred ready).
            for a in range(self._oldest()[w], start[w]):
                self._tree.set(int(self._slot(w, a)), 0.0)
            self._tree.set(int(self._slot(w, self._count[w] - 1)), 0.0)
            if stop[w] > start[w]:
                self._tree.set(int(self._slot(w, stop[w] - 1)), self._tree.max_priority)
        return workers

    def sample(self, batch_size: int, beta=0.4):
        start, stop = self._sample_bounds()
        if (stop - start).sum() <= 0 or self._tree.total() <= 0:
            raise ValueError("Cannot sample: no fully-observed n-step transitions yet")
        leaves = np.empty(batch_size, dtype=np.int64)
        segment = self._tree.total() / batch_size
        for i in range(batch_size):
            leaves[i] = self._tree.get(random.uniform(segment * i, segment * (i + 1)))
        # map flat leaf -> (worker, absolute index in that worker's valid window)
        workers, ring = np.divmod(leaves, self.capacity)
        lo = self._oldest()[workers]
        a = lo - lo % self.capacity + ring
        a = np.where(a < lo, a + self.capacity, a)
        out = self._gather(workers, a)
        priorities = self.tree_priorities(leaves)
        probs = priorities / self._tree.total()
        weights = np.power(np.maximum(len(self), 1) * probs, -beta)
        out["weights"] = (weights / weights.max()).astype(np.float32)
        out["indexes"] = leaves
        return out

    def tree_priorities(self, leaves):
        return self._tree.tree[leaves + self.max_size - 1]

    def update_priorities(self, indexes, priorities):
        p = np.power(np.asarray(priorities, dtype=np.float64) + self.eps, self.alpha)
        for leaf, pr in zip(indexes, p):
            self._tree.set(int(leaf), float(pr))
