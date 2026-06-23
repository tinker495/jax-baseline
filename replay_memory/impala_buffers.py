"""IMPALA worker-local rollout buffer (cpprb-backed).

Lives in the repo-local ``replay_memory`` adapter so the Algorithm Core's IMPALA
family reaches it only through the ``WorkerReplayBufferFactory`` seam, never by
importing cpprb directly. Unlike the generic off-policy ``ReplayBuffer`` this one
stores the behaviour-policy ``log_prob`` needed for V-trace importance sampling,
so it is a distinct class rather than a flag on the shared buffer. The shared
``batch`` record is owned by the core distributed-runtime protocol module; this
adapter depends on the core.
"""

import cpprb

from jax_baselines.core.distributed_runtime import batch


class EpochBuffer:
    def __init__(self, size: int, env_dict: dict):
        self.obsdict = {o: s for o, s in env_dict.items() if o.startswith("obs")}
        self.nextobsdict = {o: s for o, s in env_dict.items() if o.startswith("next_obs")}
        self.buffer = cpprb.ReplayBuffer(size, env_dict=env_dict)

    def __len__(self):
        return self.buffer.get_stored_size()

    def add(self, obs_t, action, log_prob, reward, nxtobs_t, terminated, truncted=False):
        obsdict = dict(zip(self.obsdict.keys(), list(obs_t)))
        nextobsdict = dict(zip(self.nextobsdict.keys(), list(nxtobs_t)))
        self.buffer.add(
            **obsdict,
            action=action,
            log_prob=log_prob,
            reward=reward,
            **nextobsdict,
            terminated=terminated,
            truncted=truncted,
        )
        if terminated or truncted:
            self.buffer.on_episode_end()

    def get_buffer(self):
        trans = self.buffer.get_all_transitions()
        return batch(
            [trans[o] for o in self.obsdict.keys()],
            trans["action"],
            trans["log_prob"],
            trans["reward"],
            [trans[o] for o in self.nextobsdict.keys()],
            trans["terminated"],
            trans["truncted"],
        )
