import numpy as np


class Buffer(object):
    def __init__(self, size: int, obs_dict: dict, env_dict: dict):
        self.max_size = size
        self._idx = 0
        self.ep_idx = 0
        self.obs_dict = obs_dict
        self.buffer = self.creat_buffer(size, obs_dict, env_dict)

    def creat_buffer(self, size: int, env_dict: dict):
        buffer = {}
        for name, data in self.obs_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        for name, data in env_dict.items():
            buffer[name] = np.zeros((size, *data["shape"]), dtype=data["dtype"])
        buffer["terminal"] = np.ones((size, 1), dtype=np.bool_)
        buffer["ep_idx"] = np.ones((size, 1), dtype=np.int32) * -1
        return buffer

    def get_stored_size(self):
        return min(self._idx, self.max_size)

    def update_idx(self):
        self._idx = self._idx + 1

    def add(self, obs, next_obs, **kwargs):
        if self.buffer["ep_idx"][self.roll_idx_m1] != self.ep_idx:
            for name, k in enumerate(self.obs_dict.keys()):
                self.buffer[name][self.roll_idx] = obs[k]
        for name, k in enumerate(self.obs_dict.keys()):
            self.buffer[name][self.next_roll_idx] = next_obs[k]
        for name, data in kwargs.items():
            self.buffer[name][self.roll_idx] = data
        self.buffer["terminal"][self.roll_idx] = 0
        self.buffer["ep_idx"][self.roll_idx] = self.ep_idx
        self.update_idx()

    def on_episode_end(self, obs, terminal):
        if not terminal:
            self.update_idx()
            for name in self.obs_dict.keys():
                self.buffer[name][self.roll_idx] = obs[name]
            self.buffer["terminal"][self.roll_idx] = 1
            self.update_idx()
        self.ep_idx += 1

    def sample_idxs(self, batch_size):
        idxs = np.random.randint(0, self.get_stored_size(), size=batch_size)
        return idxs

    @property
    def roll_idx_m1(self):
        return (self._idx - 1) % self.max_size

    @property
    def next_roll_idx(self):
        return (self._idx + 1) % self.max_size

    @property
    def roll_idx(self):
        return self._idx % self.max_size


class TransitionReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        prediction_depth=5,
    ):
        self.max_size = size
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {
                    "shape": o,
                    "dtype": np.uint8 if len(o) >= 3 else np.float32,
                },
            )
            for idx, o in enumerate(observation_space)
        )

        if isinstance(action_space, int):
            action_space = (action_space,)

        self.buffer = Buffer(
            size,
            obs_dict=self.obsdict,
            env_dict={
                "action": {"shape": action_space, "dtype": np.float32},
                "reward": {"shape": (1,), "dtype": np.float32},
            },
        )

    def __len__(self) -> int:
        return self.buffer.get_stored_size()

    @property
    def storage(self):
        return self.buffer

    @property
    def buffer_size(self) -> int:
        return self.max_size

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.max_size

    def add(self, obs_t, action, reward, nxtobs_t, terminal, truncated=False):
        pass

    def sample(self, batch_size: int):
        pass
