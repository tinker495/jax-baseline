import random

import numpy as np


class Buffer:
    def __init__(self, size: int, obs_dict: dict, env_dict: dict):
        self.max_size = size
        self._idx = -1
        self.ep_idx = 0
        self.obs_dict = obs_dict
        self.env_dict = env_dict
        self.buffer = self.create_buffer(size, obs_dict, env_dict)

    def create_buffer(self, size: int, obs_dict: dict, env_dict: dict):
        buffer = {}
        for name, data in obs_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        for name, data in env_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        buffer["terminated"] = np.ones((size, 1), dtype=np.bool_)
        buffer["ep_idx"] = np.ones((size, 1), dtype=np.int32) * -1
        return buffer

    def get_stored_size(self):
        return min(self._idx + 1, self.max_size)

    def update_idx(self):
        self._idx = self._idx + 1

    def add(self, obs, next_obs, **kwargs):
        self.update_idx()
        if self.buffer["ep_idx"][self.roll_idx_m1] != self.ep_idx:
            for idx, k in enumerate(self.obs_dict.keys()):
                self.buffer[k][self.roll_idx] = obs[idx]
        for idx, k in enumerate(self.obs_dict.keys()):
            self.buffer[k][self.next_roll_idx] = next_obs[idx]
        for k, data in kwargs.items():
            self.buffer[k][self.roll_idx] = data
        self.buffer["ep_idx"][self.roll_idx] = self.ep_idx
        return self.roll_idx

    def on_episode_end(self, truncated):
        if truncated:
            self.update_idx()
            self.buffer["ep_idx"][self.roll_idx] = -1
        self.ep_idx += 1

    def sample(self, idxs, traj_len=5):
        is_last = np.equal(self.buffer["ep_idx"][idxs], -1)[..., 0]
        idxs = np.where(is_last, idxs - 1, idxs)
        idxs = np.expand_dims(idxs, axis=1)
        obs_traj_idxs = (
            idxs + np.reshape(np.arange(traj_len + 1), (1, traj_len + 1))
        ) % self.max_size
        traj_idxs = (idxs + np.reshape(np.arange(traj_len), (1, traj_len))) % self.max_size
        obs = []
        for k in self.obs_dict:
            obs.append(self.buffer[k][obs_traj_idxs])
        data = {}
        for k in self.env_dict:
            data[k] = self.buffer[k][traj_idxs]
        terminated = self.buffer["terminated"][traj_idxs, 0]
        filled = np.equal(self.buffer["ep_idx"][idxs], self.buffer["ep_idx"][traj_idxs])[..., 0]
        return obs, data, terminated, filled

    @property
    def roll_idx_m1(self):
        return (self._idx - 1) % self.max_size

    @property
    def next_roll_idx(self):
        return (self._idx + 1) % self.max_size

    @property
    def roll_idx(self):
        return self._idx % self.max_size


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=np.int32)
        self.n_entries = 0
        self.max_priority = 1.0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # all sum of priorities
        return self.tree[0]

    def max(self):
        # return the max priority
        return self.max_priority

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        self.max_priority = max(p, self.max_priority)
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class TransitionReplayBuffer:
    def __init__(
        self,
        size: int,
        observation_space: list | None = None,
        action_space=1,
        prediction_depth=5,
    ):
        self.max_size = size
        self.prediction_depth = prediction_depth
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {
                    "shape": o,
                    "dtype": np.uint8 if len(o) >= 3 else np.float32,
                },
            )
            for idx, o in enumerate(observation_space or [])
        )

        if isinstance(action_space, int):
            action_space = (action_space,)

        self.buffer = Buffer(
            size,
            obs_dict=self.obsdict,
            env_dict={
                "actions": {"shape": action_space, "dtype": np.float32},
                "rewards": {"shape": (), "dtype": np.float32},
            },
        )

    def __len__(self) -> int:
        return self.buffer.get_stored_size()

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        self.buffer.add(
            obs_t,
            nxtobs_t,
            actions=action,
            rewards=reward,
            terminated=terminated,
        )
        if terminated or truncated:
            self.buffer.on_episode_end(truncated)

    def sample(self, batch_size: int):
        stored_size = self.buffer.get_stored_size()
        if stored_size == 0:
            raise ValueError("Cannot sample from empty buffer")
        idxs = np.random.randint(0, stored_size, size=batch_size)
        obs, data, terminated, filled = self.buffer.sample(idxs, traj_len=self.prediction_depth)
        return {"obses": obs, **data, "terminateds": terminated, "filled": filled}


class PrioritizedTransitionReplayBuffer(TransitionReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list | None = None,
        action_space=1,
        prediction_depth=5,
        alpha: float = 0.6,
        eps: float = 1e-4,
    ):
        super().__init__(size, observation_space, action_space, prediction_depth)
        self.tree = SumTree(size)
        self.alpha = alpha
        self.eps = eps

    def __len__(self) -> int:
        return self.tree.n_entries

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        idx = self.buffer.add(
            obs_t,
            nxtobs_t,
            actions=action,
            rewards=reward,
            terminated=terminated,
        )
        self.tree.add(self.tree.max(), idx)
        if terminated or truncated:
            self.buffer.on_episode_end(truncated)

    def sample(self, batch_size: int, beta=0.4):
        idxs = np.zeros((batch_size), dtype=np.int32)
        priorities = np.zeros((batch_size), dtype=np.float32)
        buffer_idxs = np.zeros((batch_size), dtype=np.int32)
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, buffer_idx = self.tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            buffer_idxs[i] = buffer_idx
        obs, data, terminated, filled = self.buffer.sample(
            buffer_idxs, traj_len=self.prediction_depth
        )
        weight = np.power(self.tree.n_entries * priorities / self.tree.total(), -beta)
        weight_max = np.max(weight)
        weights = weight / weight_max
        return {
            "obses": obs,
            **data,
            "terminateds": terminated,
            "filled": filled,
            "weights": weights,
            "indexes": idxs,
        }

    def update_priorities(self, indexes, priorities):
        priorities = np.power(priorities + self.eps, self.alpha)
        for idx, p in zip(indexes, priorities):
            self.tree.update(idx, p)
