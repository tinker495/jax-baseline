import random
from typing import Optional, List, Union

import numpy as np
import cpprb


class ReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list,
        worker_size=1,
        action_space=1,
        n_step=1,
        gamma=0.99,
    ):
        self.max_size = size
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {"shape": o, "dtype": np.uint8} if len(o) >= 3 else {"shape": o},
            )
            for idx, o in enumerate(observation_space)
        )
        self.nextobsdict = dict(
            (
                "nextobs{}".format(idx),
                {"shape": o, "dtype": np.uint8} if len(o) >= 3 else {"shape": o},
            )
            for idx, o in enumerate(observation_space)
        )
        self.n_step = n_step > 1
        n_s = None
        if self.n_step:
            n_s = {"size": n_step, "rew": "reward", "gamma": gamma, "next": "nextobs0"}
        self.buffer = cpprb.ReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": action_space},
                "reward": {},
                **self.nextobsdict,
                "done": {},
            },
            Nstep=n_s,
        )

    def __len__(self) -> int:
        return len(self.buffer)

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

    def add(self, obs_t, action, reward, nxtobs_t, done, terminal=False):
        obsdict = dict(zip(self.obsdict.keys(), obs_t))
        nextobsdict = dict(zip(self.nextobsdict.keys(), nxtobs_t))
        self.buffer.add(**obsdict, action=action, reward=reward, **nextobsdict, done=done)
        if self.n_step and terminal:
            self.buffer.on_episode_end()

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "dones": smpl["done"],
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        alpha: float,
        worker_size=1,
        action_space=1,
        n_step=1,
        gamma=0.99,
    ):
        self.max_size = size
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                {"shape": o, "dtype": np.uint8} if len(o) >= 3 else {"shape": o},
            )
            for idx, o in enumerate(observation_space)
        )
        self.nextobsdict = dict(
            (
                "nextobs{}".format(idx),
                {"shape": o, "dtype": np.uint8} if len(o) >= 3 else {"shape": o},
            )
            for idx, o in enumerate(observation_space)
        )
        self.n_step = n_step > 1
        n_s = None
        if self.n_step:
            n_s = {
                "size": n_step,
                "rew": "reward",
                "gamma": gamma,
                "next": list(self.nextobsdict.keys())[0],
            }
        self.buffer = cpprb.PrioritizedReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": action_space},
                "reward": {},
                **self.nextobsdict,
                "done": {},
            },
            alpha=alpha,
            Nstep=n_s,
        )

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "dones": smpl["done"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)
