import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import multiprocessing as mp
import time
import ray
import os

from tqdm.auto import trange
from collections import deque

from haiku_baselines.common.base_classes import (
    TensorboardWriter,
    save,
    restore,
    select_optimizer,
)
from haiku_baselines.common.utils import convert_states
from haiku_baselines.IMPALA.worker import Impala_Worker
from haiku_baselines.IMPALA.cpprb_buffers import ImpalaReplayBuffer
from haiku_baselines.common.utils import (
    convert_jax,
    discount_with_terminal,
    print_param,
)

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym import spaces


@ray.remote(num_gpus=1)
class Impala_Trainer(object):
    def __init__(
        self,
        learning_rate,
        minibatch_size,
        replay=False,
        batch_size=1024,
        network_builder=None,
        train_builder=None,
        optimizer="rmsprop",
        logger_server=None,
        key=42,
    ):
        self.buffer = ImpalaReplayBuffer(replay)
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.key_seq = hk.PRNGSequence(key)
        self.preproc, self.model = network_builder()
        self._preprocess, self._train_step = train_builder()
        self.optimizer = select_optimizer(optimizer, learning_rate, 1e-2 / batch_size)
        self.logger_server = logger_server

    def setup_model(self):
        pre_param = self.preproc.init(
            next(self.key_seq),
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        model_param = self.model.init(
            next(self.key_seq),
            self.preproc.apply(
                pre_param,
                None,
                [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
            ),
        )
        self.params = hk.data_structures.merge(pre_param, model_param)

        self.opt_state = self.optimizer.init(self.params)

        self._preprocess = jax.jit(self._preprocess)
        self._train_step = jax.jit(self._train_step)

    def get_params(self):
        return self.params

    def append_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self):
        if self.replay is not None:
            data = self.buffer.sample(self.batch_size)
        else:
            data = self.buffer
            data = {k: np.array([data[k]]) for k in data.keys()}
        self.params, loss = self._train_step(self.params, self.opt_state, next(self.key_seq))
        self.logger_server.append_loss.remote(loss)
