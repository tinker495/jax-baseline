from __future__ import annotations

import multiprocessing as mp
import random
from collections import deque
from importlib import import_module

import numpy as np

from jax_baselines.APE_X.common_servers import Logger_server, Param_server
from jax_baselines.core.seeding import seed_prngs
from jax_baselines.IMPALA.vtrace_queue import batch


def _ray():
    return import_module("ray")


class _RayWorkerHandle:
    def __init__(self, actor):
        self._actor = actor

    def get_info(self):
        return _ray().get(self._actor.get_info.remote())

    def run(self, *args, **kwargs):
        return self._actor.run.remote(*args, **kwargs)


class _RayParamServerHandle:
    def __init__(self, actor):
        self._actor = actor

    def get_params(self):
        return _ray().get(self._actor.get_params.remote())

    def update_params(self, params):
        return self._actor.update_params.remote(params)


class _RayLoggerServerHandle:
    def __init__(self, actor):
        self._actor = actor

    def get_log_dir(self):
        return _ray().get(self._actor.get_log_dir.remote())

    def add_multiline(self, eps):
        return self._actor.add_multiline.remote(eps)

    def log_trainer(self, step, log_dict):
        return self._actor.log_trainer.remote(step, log_dict)

    def log_worker(self, log_dict, episode):
        return self._actor.log_worker.remote(log_dict, episode)

    def last_update(self):
        return self._actor.last_update.remote()

    def close(self):
        return _ray().get(self._actor.close.remote())

    def register_hparams(self, hparams):
        return self._actor.register_hparams.remote(hparams)


class _ImpalaBufferGetter:
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
        return batch(*zip(*gets))

    def replay_sample(self):
        while True:
            self.replay_buffer.extend(self.queue.get_nowait_batch(self.queue.size()))
            if len(self.replay_buffer) >= self.sample_size:
                break
        gets = random.sample(self.replay_buffer, self.sample_size)
        return batch(*zip(*gets))


class RayImpalaBuffer:
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
        self.obsdict = {
            f"obs{idx}": (
                {"shape": o, "dtype": np.uint8}
                if len(o) >= 3
                else {"shape": o, "dtype": np.float32}
            )
            for idx, o in enumerate(observation_space)
        }
        self.nextobsdict = {
            f"next_obs{idx}": (
                {"shape": o, "dtype": np.uint8}
                if len(o) >= 3
                else {"shape": o, "dtype": np.float32}
            )
            for idx, o in enumerate(observation_space)
        }
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
        self.getter = (
            _ray()
            .remote(num_cpus=1)(_ImpalaBufferGetter)
            .remote(self.queue, replay_size, sample_size, getter_seed)
        )
        self.get = self.getter.sample.remote()

    def queue_info(self):
        return self.queue, self.env_dict, self.actor_num

    def __len__(self):
        return len(self.queue)

    def queue_is_empty(self):
        return self.queue.empty()

    def sample(self):
        out = _ray().get(self.get)
        self.get = self.getter.sample.remote()
        return out

    def clear(self):
        while not self.queue.empty():
            self.queue.get()


class RayDistributedRuntime:
    def __init__(self, *, num_cpus=None, num_gpus=0):
        _ray().init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
        self._manager = mp.get_context().Manager()

    def replay_manager(self):
        return self._manager

    def create_event(self):
        return self._manager.Event()

    def create_worker(self, worker_cls, *args, **kwargs):
        actor = _ray().remote(num_cpus=1)(worker_cls).remote(*args, **kwargs)
        return _RayWorkerHandle(actor)

    def worker_info(self, worker):
        return worker.get_info()

    def create_param_server(self, params):
        actor = _ray().remote(Param_server).remote(params)
        return _RayParamServerHandle(actor)

    def create_logger_server(
        self, log_dir, run_name, experiment_name="experiment", logger_factory=None
    ):
        actor = (
            _ray().remote(Logger_server).remote(log_dir, run_name, experiment_name, logger_factory)
        )
        return _RayLoggerServerHandle(actor)

    def create_impala_buffer(self, *args, **kwargs):
        return RayImpalaBuffer(*args, **kwargs)

    def wait(self, jobs, timeout=None):
        return _ray().wait(jobs, timeout=timeout)

    def shutdown(self):
        self._manager.shutdown()
