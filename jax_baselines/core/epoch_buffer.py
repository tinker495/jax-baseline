"""Runtime-neutral on-policy epoch buffer for actor-critic rollouts."""

from __future__ import annotations

import numpy as np


def _as_array(values):
    return np.asarray(values)


class EpochBuffer(object):
    def __init__(self, epoch_size: int, observation_space: list, worker_size=1, action_space=1):
        self.epoch_size = epoch_size
        self.observation_space = observation_space
        self.worker_size = worker_size
        self.action_space = action_space
        self.local_buffers = [[] for _ in range(worker_size)]

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated):
        for worker_idx in range(self.worker_size):
            self.local_buffers[worker_idx].append(
                {
                    "obses": [np.asarray(obs[worker_idx]) for obs in obs_t],
                    "action": np.asarray(action[worker_idx]),
                    "reward": np.asarray(reward[worker_idx]),
                    "nxtobses": [np.asarray(next_obs[worker_idx]) for next_obs in nxtobs_t],
                    "terminated": np.asarray(terminated[worker_idx]),
                    "truncated": np.asarray(truncated[worker_idx]),
                }
            )

    def get_buffer(self):
        transitions = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "nxtobses": [],
            "terminateds": [],
            "truncateds": [],
        }
        for worker_buffer in self.local_buffers:
            transitions["obses"].append(
                [
                    _as_array([record["obses"][obs_idx] for record in worker_buffer])
                    for obs_idx in range(len(self.observation_space))
                ]
            )
            transitions["actions"].append(_as_array([record["action"] for record in worker_buffer]))
            transitions["rewards"].append(_as_array([record["reward"] for record in worker_buffer]))
            transitions["nxtobses"].append(
                [
                    _as_array([record["nxtobses"][obs_idx] for record in worker_buffer])
                    for obs_idx in range(len(self.observation_space))
                ]
            )
            transitions["terminateds"].append(
                _as_array([record["terminated"] for record in worker_buffer])
            )
            transitions["truncateds"].append(
                _as_array([record["truncated"] for record in worker_buffer])
            )
            worker_buffer.clear()
        return transitions
