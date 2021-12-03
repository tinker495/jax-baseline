import os
import argparse
import gym

import jax
import jax.numpy as jnp
import numpy as np
import ray

from abc import ABC

class Multiworker(ABC):
    def __init__(self,env_id, worker_num = 8):
        pass
        
    def step(self,actions):
        pass
        
    def get_steps(self):
        pass

class gymMultiworker(Multiworker):
    def __init__(self,env_id, worker_num = 8):
        self.env_id = env_id
        self.worker_num = worker_num
        self.workers = [gymRayworker.remote(env_id) for _ in range(worker_num)]
        self.env_info = ray.get(self.workers[0].get_info.remote())
        self.steps = [w.get_reset.remote() for w in self.workers]
        
    def step(self,actions):
        self.steps = [w.step.remote(a) for w,a in zip(self.workers,actions)]
        
    def get_steps(self):
        states = []; rewards =[]; dones = []; terminals = []; end_states = []; end_idx = [];
        for idx,data in enumerate(ray.get(self.steps)):
            state, end_state, reward, done, terminal = data
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            terminals.append(terminal)
            if end_states is not None:
                end_states.append(end_state)
                end_idx.append(idx)
        states = np.stack(states,axis=0)
        rewards = np.stack(rewards,axis=0)
        dones = np.stack(dones,axis=0)
        terminals = np.stack(terminals,axis=0)
        if len(end_states) > 0:
            end_states = np.stack(end_states,axis=0)
            end_idx = np.stack(end_idx,axis=0)
        else:
            end_states = None
            end_idx = None
        return states,rewards,dones,terminals,end_states,end_idx

@ray.remote
class gymRayworker:
    def __init__(self, env_name_):
        from haiku_baselines.common.atari_wrappers import make_wrap_atari,get_env_type
        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == 'atari':
            self.env = make_wrap_atari(self.env_id)
        else:
            self.env = gym.make(self.env_id)
        
    def get_reset(self):
        return self.env.reset(),None,0,False,False
    
    def get_info(self):
        return {'observation_space' : self.env.observation_space, 
                'action_space' : self.env.action_space,
                'env_type' : self.env_type,
                'env_id' : self.env_id}
        
    def step(self,action):
        state, reward, terminal, info = self.env.step(action)
        done = terminal
        if "TimeLimit.truncated" in info:
            done = not info["TimeLimit.truncated"]
        if terminal:
            end_state = state
            state = self.env.reset()
        else:
            end_state = None
        return state, end_state, reward, done, terminal