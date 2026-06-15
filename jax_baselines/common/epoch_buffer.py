"""On-policy epoch replay buffer kept under :mod:`jax_baselines.common`."""

import cpprb
import numpy as np


def _obs_spec(shape):
    """uint8 spec for image-like (>=3 dims) features, float32 otherwise."""
    return (
        {"shape": shape, "dtype": np.uint8}
        if len(shape) >= 3
        else {"shape": shape, "dtype": np.float32}
    )


def _build_obs_dicts(observation_space):
    """Build the ``obs{i}`` / ``next_obs{i}`` cpprb env_dict fragments."""
    obsdict = {f"obs{idx}": _obs_spec(o) for idx, o in enumerate(observation_space)}
    nextobsdict = {f"next_obs{idx}": _obs_spec(o) for idx, o in enumerate(observation_space)}
    return obsdict, nextobsdict


class EpochBuffer(object):
    def __init__(self, epoch_size: int, observation_space: list, worker_size=1, action_space=1):
        self.epoch_size = epoch_size
        self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
        self.worker_size = worker_size
        self.local_buffers = [
            cpprb.ReplayBuffer(
                epoch_size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                    "truncated": {},
                },
            )
            for _ in range(worker_size)
        ]

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated):
        for w in range(self.worker_size):
            obsdict = dict(zip(self.obsdict, [o[w] for o in obs_t]))
            nextobsdict = dict(zip(self.nextobsdict, [no[w] for no in nxtobs_t]))
            self.local_buffers[w].add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                terminated=terminated[w],
                truncated=truncated[w],
            )
            if terminated[w] or truncated[w]:
                self.local_buffers[w].on_episode_end()

    def get_buffer(self):
        transitions = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "nxtobses": [],
            "terminateds": [],
            "truncateds": [],
        }
        for w in range(self.worker_size):
            trans = self.local_buffers[w].get_all_transitions()
            transitions["obses"].append([trans[o] for o in self.obsdict])
            transitions["actions"].append(trans["action"])
            transitions["rewards"].append(trans["reward"])
            transitions["nxtobses"].append([trans[o] for o in self.nextobsdict])
            transitions["terminateds"].append(trans["terminated"])
            transitions["truncateds"].append(trans["truncated"])
            self.local_buffers[w].clear()
        return transitions
