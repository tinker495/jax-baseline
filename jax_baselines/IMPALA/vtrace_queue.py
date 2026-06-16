"""IMPALA V-trace rollout queue (Ray-backed shared replay).

Distributed *runtime wiring* that the Algorithm Core keeps as a consciously
declared Ray exemption (ADR 0002 ``DISTRIBUTED_RUNTIME_EXEMPTION``): the cross-actor
queue and its getter are Ray actors. The cpprb-backed worker-local rollout buffer
(``EpochBuffer``) lives in the ``replay_memory`` adapter and reaches the core only
through the ``WorkerReplayBufferFactory`` seam; the shared ``batch`` record is
defined here so the adapter depends on the core, never the reverse.
"""

import random
from collections import deque, namedtuple
from importlib import import_module

import numpy as np

from jax_baselines.core.seeding import seed_prngs

batch = namedtuple(
    "batch_tuple",
    ["obses", "actions", "mu_log_prob", "rewards", "nxtobses", "terminateds", "truncateds"],
)


class Buffer_getter:
    @classmethod
    def remote(cls, *args, **kwargs):
        return import_module("ray").remote(num_cpus=1)(cls).remote(*args, **kwargs)

    def __init__(self, queue, size, sample_size, seed=None):
        self.queue = queue
        self.replay = size > 0
        self.sample_size = sample_size
        seed_prngs(seed)
        if self.replay:
            self.replay_buffer = deque(maxlen=size)
            self._sample = self.replay_sample
        else:
            self._sample = self.queue_sample

    def sample(self):
        return self._sample()

    def queue_sample(self):
        gets = [self.queue.get() for _ in range(self.sample_size)]
        transitions = batch(*zip(*gets))
        return transitions

    def replay_sample(self):
        while True:
            self.replay_buffer.extend(self.queue.get_nowait_batch(self.queue.size()))
            if len(self.replay_buffer) >= self.sample_size:
                break
        gets = random.sample(self.replay_buffer, self.sample_size)
        transitions = batch(*zip(*gets))
        return transitions


class ImpalaBuffer:
    def __init__(
        self,
        replay_size: int,
        actor_num: int,
        observation_space: list,
        discrete=True,
        action_space=1,
        sample_size=32,
        seed=None,
    ):
        self.actor_num = actor_num
        self.obsdict = dict(
            (
                "obs{}".format(idx),
                (
                    {"shape": o, "dtype": np.uint8}
                    if len(o) >= 3
                    else {"shape": o, "dtype": np.float32}
                ),
            )
            for idx, o in enumerate(observation_space)
        )
        self.nextobsdict = dict(
            (
                "next_obs{}".format(idx),
                (
                    {"shape": o, "dtype": np.uint8}
                    if len(o) >= 3
                    else {"shape": o, "dtype": np.float32}
                ),
            )
            for idx, o in enumerate(observation_space)
        )

        self.env_dict = {
            **self.obsdict,
            "action": {"shape": 1 if discrete else action_space},
            "log_prob": {},
            "reward": {},
            **self.nextobsdict,
            "terminated": {},
            "truncted": {},
        }

        queue_type = import_module("ray.util.queue").Queue
        self.queue = queue_type(maxsize=max(actor_num * 2, replay_size))
        getter_seed = None if seed is None else seed + 10_000
        self.getter = Buffer_getter.remote(self.queue, replay_size, sample_size, getter_seed)
        self.get = self.getter.sample.remote()

    def queue_info(self):
        return self.queue, self.env_dict, self.actor_num

    def __len__(self):
        return len(self.queue)

    def queue_is_empty(self):
        return self.queue.empty()

    def sample(self):
        out = import_module("ray").get(self.get)
        self.get = self.getter.sample.remote()
        return out
