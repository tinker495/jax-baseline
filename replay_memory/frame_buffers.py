"""Frame-level transition replay buffer for image / frame-stacked observations.

cpprb's ``stack_compress`` cannot compact the n-step ``next_obs`` (its rows are n
steps apart, so the sliding-window reconstruction breaks — verified empirically).
This buffer instead stores a single newest frame per observation and reconstructs
both the frame-stack and the n-step next observation by index, so an n-step Atari
replay at 1e6 costs ~7GB instead of ~35GB (full ``next_obs``).

Design (dopamine ``OutOfGraphReplayBuffer`` style):
  * store one newest frame per transition; reconstruct an S-frame stack by gathering
    S consecutive frames, padding at the episode start (matching gym ``FrameStack``,
    which repeats the reset frame and keeps the newest frame in the last channel).
  * compute the n-step reward / next index / done at sample time from per-step
    reward+terminated, bounded by the episode (``ep_id`` / ``ep_step``).

Scope: single image modality, single worker (``worker_size==1``). The replay
factory falls back to cpprb for vector obs, multi-modal obs, or vectorised envs.
"""

import random

import numpy as np


def _frame_geometry(observation_space, n_frames):
    shape = next(iter(observation_space.values()), ())
    if len(observation_space) != 1 or len(shape) < 3:
        raise ValueError(
            "FrameStackReplayBuffer supports a single image modality "
            f"(H, W, C*stack); got observation_space={observation_space}"
        )
    h, w, stacked_c = shape
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
    ):
        self.max_size = int(size)
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
        self._ep_id = np.full((self.max_size,), -1, dtype=np.int64)
        self._ep_step = np.zeros((self.max_size,), dtype=np.int32)
        self._boundary_next = {}

        self._count = 0  # total transitions ever added (monotonic)
        self._ep = 0
        self._cur_step = 0
        self._discounts = (gamma ** np.arange(self.n_step)).astype(np.float64)

    # ---- bookkeeping ----------------------------------------------------
    def __len__(self) -> int:
        return min(self._count, self.max_size)

    def _ready(self) -> int:
        # transitions whose full n-step window is observable: exclude the newest
        # (n_step-1) that may still be growing toward a boundary or full window.
        return max(0, min(self._count, self.max_size) - (self.n_step - 1) - 1)

    def episode_end(self):
        # add() already advances the episode on terminated/truncated; kept for
        # interface parity with the cpprb buffers.
        pass

    # ---- add ------------------------------------------------------------
    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        r = self._count % self.max_size
        self._boundary_next.pop(r, None)
        self._frame[r] = np.asarray(obs_t[self.observation_key])[0, :, :, -self.cf :]
        self._action[r] = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        self._reward[r] = reward
        self._terminated[r] = bool(terminated)
        self._truncated[r] = bool(truncated)
        self._ep_id[r] = self._ep
        self._ep_step[r] = self._cur_step
        if terminated or truncated:
            self._boundary_next[r] = np.asarray(nxtobs_t[self.observation_key])[
                0, :, :, -self.cf :
            ].copy()
        self._count += 1
        if terminated or truncated:
            self._ep += 1
            self._cur_step = 0
        else:
            self._cur_step += 1

    # ---- reconstruction -------------------------------------------------
    def _gather_stack(self, abs_idx, lo):
        """Reconstruct the S-frame stacked observation for absolute indices.

        abs_idx, lo: int arrays of shape (B,). Frames are stacked oldest-first
        along the last axis (newest in the final channel block), padded at the
        episode start by repeating the first in-episode frame.
        """
        b = abs_idx.shape[0]
        e = self._ep_step[abs_idx % self.max_size]  # steps available before this obs
        out = np.empty((b, self.h, self.w, self.stacked_c), dtype=np.uint8)
        for back in range(self.n_frames):  # back=0 -> newest
            off = np.minimum(np.minimum(back, e), abs_idx - lo)
            src = (abs_idx - off) % self.max_size
            ch = self.n_frames - 1 - back
            out[..., ch * self.cf : (ch + 1) * self.cf] = self._frame[src]
        return out

    def _nstep(self, a):
        """Vectorised n-step over absolute start indices a (B,).

        Returns (reward (B,), next_idx (B,), done (B,), boundary (B,))."""
        b = a.shape[0]
        reward = np.zeros(b, dtype=np.float64)
        steps = np.zeros(b, dtype=np.int64)
        done = np.zeros(b, dtype=np.float32)
        active = np.ones(b, dtype=bool)
        for k in range(self.n_step):
            idx = (a + k) % self.max_size
            reward += active * self._discounts[k] * self._reward[idx]
            steps += active
            term = self._terminated[idx] & active
            trunc = self._truncated[idx] & active
            done = np.where(term, 1.0, done)
            active = active & ~(term | trunc)
        hit_boundary = ~active
        next_idx = a + steps - hit_boundary.astype(np.int64)
        return reward.astype(np.float32), next_idx, done, hit_boundary

    def _gather(self, a):
        lo = max(0, self._count - self.max_size)
        lo_arr = np.full(a.shape, lo, dtype=np.int64)
        reward, next_idx, done, hit_boundary = self._nstep(a)
        obses = {self.observation_key: self._gather_stack(a, lo_arr)}
        next_stack = self._gather_stack(next_idx, lo_arr)
        for row in np.flatnonzero(hit_boundary):
            next_stack[row, ..., : -self.cf] = next_stack[row, ..., self.cf :]
            next_stack[row, ..., -self.cf :] = self._boundary_next[next_idx[row] % self.max_size]
        nxtobses = {self.observation_key: next_stack}
        return {
            "obses": obses,
            "actions": self._action[a % self.max_size],
            "rewards": reward[:, None],
            "nxtobses": nxtobses,
            "terminateds": done[:, None],
        }

    def _sample_indices(self, batch_size):
        lo = max(0, self._count - self.max_size)
        hi = lo + self._ready()  # exclusive upper bound of fully-observable starts
        return np.random.randint(lo, hi, size=batch_size).astype(np.int64)

    def sample(self, batch_size: int):
        if self._ready() <= 0:
            raise ValueError("Cannot sample: no fully-observed n-step transitions yet")
        return self._gather(self._sample_indices(batch_size))


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
    ):
        super().__init__(size, observation_space, action_space, n_step, gamma, n_frames)
        self.alpha = alpha
        self.eps = eps
        self._tree = _SumTree(self.max_size)

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        super().add(obs_t, action, reward, nxtobs_t, terminated, truncated)
        # the just-added slot is not sampleable yet (its n-step window is unobserved)
        self._tree.set((self._count - 1) % self.max_size, 0.0)
        # a transition becomes sampleable once frame[a + n_step] exists, i.e. n_step+1
        # adds later; assign it max priority then (deferred ready).
        ready_abs = self._count - self.n_step - 1
        if ready_abs >= max(0, self._count - self.max_size):
            self._tree.set(ready_abs % self.max_size, self._tree.max_priority)

    def sample(self, batch_size: int, beta=0.4):
        if self._ready() <= 0 or self._tree.total() <= 0:
            raise ValueError("Cannot sample: no fully-observed n-step transitions yet")
        leaves = np.empty(batch_size, dtype=np.int64)
        segment = self._tree.total() / batch_size
        for i in range(batch_size):
            leaves[i] = self._tree.get(random.uniform(segment * i, segment * (i + 1)))
        # map ring leaf -> absolute index in the current valid window
        lo = max(0, self._count - self.max_size)
        base = lo - (lo % self.max_size)
        a = base + leaves
        a = np.where(a < lo, a + self.max_size, a)
        out = self._gather(a)
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
