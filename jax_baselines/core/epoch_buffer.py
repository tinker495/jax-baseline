"""Runtime-neutral on-policy epoch buffer for actor-critic rollouts."""

from __future__ import annotations

import numpy as np


class EpochBuffer:
    def __init__(self, epoch_size: int, observation_space: dict, worker_size=1, action_space=1):
        self.epoch_size = epoch_size
        self.observation_space = observation_space
        self.worker_size = worker_size
        self.action_space = action_space
        self.local_buffers = [[] for _ in range(worker_size)]

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated):
        for worker_idx in range(self.worker_size):
            self.local_buffers[worker_idx].append(
                {
                    "obses": {
                        key: np.asarray(obs_t[key][worker_idx]) for key in self.observation_space
                    },
                    "action": np.asarray(action[worker_idx]),
                    "reward": np.asarray(reward[worker_idx]),
                    "nxtobses": {
                        key: np.asarray(nxtobs_t[key][worker_idx]) for key in self.observation_space
                    },
                    "terminated": np.asarray(terminated[worker_idx]),
                    "truncated": np.asarray(truncated[worker_idx]),
                }
            )

    def get_buffer(self):
        transitions = {
            "obses": {},
            "actions": [],
            "rewards": [],
            "nxtobses": {},
            "terminateds": [],
            "truncateds": [],
        }
        for key in self.observation_space:
            transitions["obses"][key] = np.asarray(
                [
                    [record["obses"][key] for record in worker_buffer]
                    for worker_buffer in self.local_buffers
                ]
            )
            transitions["nxtobses"][key] = np.asarray(
                [
                    [record["nxtobses"][key] for record in worker_buffer]
                    for worker_buffer in self.local_buffers
                ]
            )
        for worker_buffer in self.local_buffers:
            transitions["actions"].append(
                np.asarray([record["action"] for record in worker_buffer])
            )
            transitions["rewards"].append(
                np.asarray([record["reward"] for record in worker_buffer])
            )
            transitions["terminateds"].append(
                np.asarray([record["terminated"] for record in worker_buffer])
            )
            transitions["truncateds"].append(
                np.asarray([record["truncated"] for record in worker_buffer])
            )
            worker_buffer.clear()
        return transitions
