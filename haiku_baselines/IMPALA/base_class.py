import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import os

from tqdm.auto import trange
from collections import deque

from haiku_baselines.common.base_classes import TensorboardWriter, save, restore, select_optimizer
from haiku_baselines.common.buffers import EpochBuffer
from haiku_baselines.common.utils import convert_states
from haiku_baselines.common.worker import gymMultiworker
from haiku_baselines.common.utils import convert_jax, discount_with_terminal, print_param

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym import spaces
import ray

class IMPALA_base(object):
    def __init__(self, env_name, worker_size=16, worker_id=0, time_scale=20, capture_frame_rate=1, **kwargs):
        self.env_name = env_name
        
        if os.path.exists(env_name):
            """Unity Environment"""
            """not implemented yet"""
        else:
            self.worker_size = worker_size
            ray.init(num_cpus=self.worker_size + 2)
            env_type = "gym"
            self.workers = [IMPALA_gym_worker.remote(env_name) for _ in range(self.worker_size)]
        
    def set_model(self, model, trainer):
        self.model = model
        self.trainer = trainer

@ray.remote
class IMPALA_gym_worker:
    def __init__(self, env_name_, model) -> None:
        from haiku_baselines.common.atari_wrappers import make_wrap_atari,get_env_type
        self.env_type, self.env_id = get_env_type(env_name_)
        if  self.env_type == 'atari_env':
            self.env = make_wrap_atari(env_name_,clip_rewards=True)
        else:
            self.env = gym.make(env_name_)

        self.action_type = 'continuous' if isinstance(self.env.action_space, spaces.Box) else 'discrete'
        if self.action_type == 'discrete':
            self.action_conv =  lambda a: a[0]
            self._get_actions = self._get_actions_discrete
            self.actions = self.action_discrete
        elif self.action_type == 'continuous':
            self.action_conv =  lambda a: a
            self._get_actions = self._get_actions_continuous
            self.actions = self.action_continuous

        self.actions = jax.jit(self._get_actions)

    def set_model(self, preproc, actor):
        self.preproc = preproc
        self.actor = actor

    def get_info(self):
        return {'observation_space' : self.env.observation_space, 
                'action_space' : self.env.action_space,
                'env_type' : self.env_type,
                'env_id' : self.env_id}
    
    def _get_actions_discrete(self, params, obses, key = None) -> jnp.ndarray:
        prob = jax.nn.softmax(self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses))),axis=1,)
        return prob
    
    def _get_actions_continuous(self, params, obses, key = None) -> jnp.ndarray:
        mu,std = self.actor.apply(params, key, self.preproc.apply(params, key, convert_jax(obses)))
        return mu, jnp.exp(std)

    def action_discrete(self,obs):
        prob = np.asarray(self._get_actions(self.params, obs))
        return np.expand_dims(np.stack([np.random.choice(self.action_size[0],p=p) for p in prob],axis=0),axis=1), prob
    
    def action_continuous(self,obs):
        mu, std = self._get_actions(self.params, obs)
        return np.random.normal(np.array(mu), np.array(std)), np.array(mu), np.array(std)
    
    def run(self, params, n_step, render=False):
        state, info = self.env.reset()
        state = [np.expand_dims(state,axis=0)]
        while True:
            action = self.action_conv(self.actions(state))
            state, reward, done, info = self.env.step(action)
            state = [np.expand_dims(state,axis=0)]
            if render:
                self.env.render()
            if done:
                break
