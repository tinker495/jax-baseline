import random
from typing import Optional, List, Union

import jax
import numpy as np
import jax.numpy as jnp

from haiku_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

@jax.jit
def discounted(rewards,gamma=0.99): #lfilter([1],[1,-gamma],x[::-1])[::-1]
    _gamma = 1
    out = 0
    for r in rewards:
        out += r*_gamma
        _gamma *= gamma
    return out

class ReplayBuffer(object):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1, n_step=1, gamma=0.99):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._maxsize = size
        self.worker_size = worker_size
        self.observation_space = observation_space
        self.action_space = action_space
        self._next_idx = 0
        self._len = 0
        self.n_step = n_step
        self.n_step_method = self.n_step > 1
        self.gamma = gamma
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": jnp.uint8} if len(o) >= 3 else {"shape": o})
                            for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o,"dtype": jnp.uint8} if len(o) >= 3 else {"shape": o})
                            for idx,o in enumerate(observation_space))
        self._storage = dict(
            [(
                key, jnp.zeros((self._maxsize,*self.obsdict[key]["shape"]),dtype=self.obsdict[key]['dtype'] if 'dtype' in self.obsdict[key] else jnp.float32)
            )   for key in self.obsdict]
            +
            [(
                'rewards', jnp.zeros((self._maxsize,1),dtype=jnp.float32)
            )
            ,
            (
                'actions', jnp.zeros((self._maxsize,*self.action_space),dtype=jnp.float32)
            )]
            +
            [(
                key, jnp.zeros((self._maxsize,*self.nextobsdict[key]["shape"]),dtype=self.nextobsdict[key]['dtype'] if 'dtype' in self.nextobsdict[key] else jnp.float32)
            )   for key in self.nextobsdict]
            +[(
                'dones', jnp.zeros((self._maxsize,1),dtype=jnp.float32)
            )]
            )
        self.episodes = None
        self.worker_ep = None
        self.worker_range = None
        if self.n_step_method:
            self.episodes = {}
            self.worker_ep = jnp.zeros(worker_size)
            self.worker_range = jnp.arange(0,worker_size)
            self._storage.update(
            dict([
            (
                'episode', jnp.zeros((self._maxsize,1),dtype=jnp.int32)
            )
            ,
            (
                'steps', jnp.zeros((self._maxsize,1),dtype=jnp.int32)
            )
            ,
            (
                'terminal', jnp.zeros((self._maxsize,1),dtype=jnp.int32)
            )
            ])
            )
        
        #self._add = jax.jit(self._add)
            
    @property
    def __len__(self) -> int:
        return self._len

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        next_idxs = jnp.arange(self._next_idx,self._next_idx+self.worker_size,dtype=jnp.int32) % self._maxsize
        self._next_idx = (self._next_idx + self.worker_size) % self._maxsize
        self._len = jnp.where(self._len + self.worker_size >= self._maxsize, self._maxsize, self._len + self.worker_size)
        episode_keys = None
        steps = None
        if self.n_step_method:
            episode_keys = jnp.array(self.worker_range + self.worker_size*self.worker_ep,dtype=jnp.int32)
            steps = jnp.zeros((self.worker_size),dtype=jnp.int32)
            for i,(ep_key,n_idx) in enumerate(zip(episode_keys,next_idxs)):
                if ep_key not in self.episodes:
                    self.episodes[ep_key] = []
                steps.at[i].set(len(self.episodes[ep_key]))
                self.episodes[ep_key].append(n_idx)
                
            for e_k,t in zip(self._storage['episode'][next_idxs],self._storage['terminal'][next_idxs]):
                if t:
                    del self.episodes[e_k]
                
        self._storage, self.worker_ep = self._add(self._storage, self.worker_ep, next_idxs, episode_keys, steps, obs_t, action, reward, nxtobs_t, done, terminal)
        
    def _add(self, storage, worker_ep, next_idxs, episode_keys, steps, obs_t, action, reward, nxtobs_t, done, terminal):
        obses_dicts = dict(zip(self.obsdict.keys(),obs_t) +\
                           zip(self.nextobsdict.keys(),nxtobs_t))
        for k,data in obses_dicts:
            storage[k].at[next_idxs].set(data)
        storage['actions'].at[next_idxs].set(action)
        storage['rewards'].at[next_idxs].set(reward)
        storage['dones'].at[next_idxs].set(done)
        if self.n_step_method:
            storage['episode'].at[next_idxs].set(episode_keys)
            storage['steps'].at[next_idxs].set(steps)
            storage['terminal'].at[next_idxs].set(terminal)
            worker_ep = jnp.where(terminal > 0, worker_ep + 1, worker_ep)
        return storage, worker_ep
    
    def _encode_sample(self, idxes: Union[List[int], jnp.ndarray]):
        if self.n_step_method:
            eps = self._storage['episode'][idxes]
            steps = self._storage['steps'][idxes]
            n_idxes = [self.episodes[ep][step:(step + self.n_step)] for ep,step in zip(eps,steps)]
            r_discounted = jnp.expand_dims(jnp.array([discounted(self._storage['rewards'][nidx],self.gamma) for nidx in n_idxes]))
            l_idxes = jnp.array([ni[-1] for ni in n_idxes])
            return {
                'obses'     : [self._storage[o][idxes] for o in self.obsdict.keys()],
                'actions'   : self._storage['actions'][idxes],
                'rewards'   : r_discounted,
                'nxtobses'  : [self._storage[no][l_idxes] for no in self.nextobsdict.keys()],
                'dones'     : self._storage['dones'][l_idxes]
                    }
        else:
            return {
                'obses'     : [self._storage[o][idxes] for o in self.obsdict.keys()],
                'actions'   : self._storage['actions'][idxes],
                'rewards'   : self._storage['rewards'][idxes],
                'nxtobses'  : [self._storage[no][idxes] for no in self.nextobsdict.keys()],
                'dones'     : self._storage['dones'][idxes]
                    }

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = np.random.randint(0, self._len - 1,(batch_size,)) #jax.random.randint(key, (batch_size), )
        return self._encode_sample(idxes)