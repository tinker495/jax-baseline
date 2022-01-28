import random
from typing import Optional, List, Union
from collections.abc import Iterable

import jax
import numpy as np
import jax.numpy as jnp

from haiku_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class EpochBuffer(object):
    def __init__(self, epoch_size : int, observation_space: list, worker_size = 1, action_space = 1):
        self._maxsize = epoch_size
        self.worker_size = worker_size
        self.observation_space = observation_space
        self.action_space = action_space
        self._next_idx = 0
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self._storage = dict(
            [(
                key, np.zeros((self.worker_size,self._maxsize,*self.obsdict[key]["shape"]),dtype=self.obsdict[key]['dtype'])
            )   for key in self.obsdict]
            +
            [(
                'rewards', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            )
            ,
            (
                'actions', np.zeros((self.worker_size,self._maxsize,*self.action_space),dtype=np.float32)
            )]
            +
            [(
                key, np.zeros((self.worker_size,self._maxsize,*self.nextobsdict[key]["shape"]),dtype=self.nextobsdict[key]['dtype'])
            )   for key in self.nextobsdict]
            +[(
                'dones', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            ),(
                'terminals', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            )
            ]
            )
        
    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        obses_dicts = dict(zip(self.obsdict.keys(),obs_t))
        nxtobses_dicts = dict(zip(self.nextobsdict.keys(),nxtobs_t))
        for k in obses_dicts:
            self._storage[k][:,self._next_idx,:] = obses_dicts[k]
        for k in nxtobses_dicts:
            self._storage[k][:,self._next_idx,:] = nxtobses_dicts[k]
        self._storage['actions'][:,self._next_idx,:] = action
        self._storage['rewards'][:,self._next_idx,:] = reward
        self._storage['dones'][:,self._next_idx,:] = done
        self._storage['terminals'][:,self._next_idx,:] = terminal
        self._next_idx += 1

    def get_buffer(self):
        return {
            'obses'     : [[self._storage[o][w,:] for o in self.obsdict.keys()] for w in range(self.worker_size)],
            'actions'   : [self._storage['actions'][w,:] for w in range(self.worker_size)],
            'rewards'   : [self._storage['rewards'][w,:] for w in range(self.worker_size)],
            'nxtobses'  : [[self._storage[no][w,:] for no in self.nextobsdict.keys()] for w in range(self.worker_size)],
            'dones'     : [self._storage['dones'][w,:] for w in range(self.worker_size)],
            'terminals'     : [self._storage['terminals'][w,:] for w in range(self.worker_size)]
            }
        
    def clear(self):
        self._next_idx = 0
        self._storage = dict(
            [(
                key, np.zeros((self.worker_size,self._maxsize,*self.obsdict[key]["shape"]),dtype=self.obsdict[key]['dtype'])
            )   for key in self.obsdict]
            +
            [(
                'rewards', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            )
            ,
            (
                'actions', np.zeros((self.worker_size,self._maxsize,*self.action_space),dtype=np.float32)
            )]
            +
            [(
                key, np.zeros((self.worker_size,self._maxsize,*self.nextobsdict[key]["shape"]),dtype=self.nextobsdict[key]['dtype'])
            )   for key in self.nextobsdict]
            +[(
                'dones', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            ),(
                'terminals', np.zeros((self.worker_size,self._maxsize,1),dtype=np.float32)
            )
            ]
            
            )

@jax.jit
def discounted(rewards,gamma=0.99): #lfilter([1],[1,-gamma],x[::-1])[::-1]
    _gamma = 1
    out = 0
    for r in rewards:
        out += r*_gamma
        _gamma *= gamma
    return out

class ReplayBuffer(object):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._maxsize = size
        self.worker_size = worker_size
        self.observation_space = observation_space
        if isinstance(action_space, Iterable):
            self.action_space = action_space
        else:
            self.action_space = [action_space]
        self._next_idx = 0
        self._len = 0
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self._storage = dict(
            [(
                key, np.zeros((self._maxsize,*self.obsdict[key]["shape"]),dtype=self.obsdict[key]['dtype'])
            )   for key in self.obsdict]
            +
            [(
                'rewards', np.zeros((self._maxsize,1),dtype=np.float32)
            )
            ,
            (
                'actions', np.zeros((self._maxsize,*self.action_space),dtype=np.float32)
            )]
            +
            [(
                key, np.zeros((self._maxsize,*self.nextobsdict[key]["shape"]),dtype=self.nextobsdict[key]['dtype'])
            )   for key in self.nextobsdict]
            +[(
                'dones', np.zeros((self._maxsize,1),dtype=np.float32)
            ),(
                'terminals', np.zeros((self._maxsize,1),dtype=np.float32)
            )
            ]
            )
            
    @property
    def __len__(self) -> int:
        return self._len

    @property
    def buffer_size(self) -> int:
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.buffer_size
    
    def _add(self, nxt_idxs, obs_t, action, reward, nxtobs_t, done, terminal):
        obses_dicts = dict(zip(self.obsdict.keys(),obs_t))
        nxtobses_dicts = dict(zip(self.nextobsdict.keys(),nxtobs_t))
        for k in obses_dicts:
            self._storage[k][nxt_idxs,:] = obses_dicts[k]
        for k in nxtobses_dicts:
            self._storage[k][nxt_idxs,:] = nxtobses_dicts[k]
        self._storage['actions'][nxt_idxs,:] = action
        self._storage['rewards'][nxt_idxs,:] = reward
        self._storage['dones'][nxt_idxs,:] = done
        self._storage['terminals'][nxt_idxs,:] = terminal
        
    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        nxt_idxs = (np.arange(self.worker_size+1) + self._next_idx) % self._maxsize
        self._next_idx = nxt_idxs[-1]; nxt_idxs = nxt_idxs[:-1]
        self._add(nxt_idxs, obs_t, action, reward, nxtobs_t, done, terminal)
        if self._next_idx > self._len:
            self._len = self._next_idx
        else:
            self._len = self._maxsize
        
    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        return {
            'obses'     : [self._storage[o][idxes] for o in self.obsdict.keys()],
            'actions'   : self._storage['actions'][idxes],
            'rewards'   : self._storage['rewards'][idxes],
            'nxtobses'  : [self._storage[no][idxes] for no in self.nextobsdict.keys()],
            'dones'     : self._storage['dones'][idxes],
            #'terminals'     : self._storage['terminals'][idxes]
            }
    
    def sample(self, batch_size: int):
        idxes = [random.randint(0, self._len - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1, alpha = 0.4):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        super(PrioritizedReplayBuffer, self).__init__(size, observation_space,worker_size,action_space)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        nxt_idxs = (np.arange(self.worker_size+1) + self._next_idx) % self._maxsize
        self._next_idx = nxt_idxs[-1]; nxt_idxs = nxt_idxs[:-1]
        self._add(nxt_idxs, obs_t, action, reward, nxtobs_t, done, terminal)
        self._it_sum[nxt_idxs] = self._max_priority ** self._alpha
        self._it_min[nxt_idxs] = self._max_priority ** self._alpha
        if self._next_idx > self._len:
            self._len = self._next_idx
        else:
            self._len = self._maxsize
    
    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, self._len - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = np.array(self._it_sum.find_prefixsum_idx(mass))
        return idx
    
    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = np.array((p_sample * len(self._storage)) ** (-beta) / max_weight)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample['weights'] = weights
        encoded_sample['indexes'] = idxes
        return encoded_sample
    
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
        
class EpisodicReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1, n_step=3, gamma=0.99):
        """
        Create Episodic Replay buffer for n-step td

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(EpisodicReplayBuffer, self).__init__(size, observation_space, worker_size, action_space)
        self._storage['episode'] = np.zeros((self._maxsize, 3),np.int32)
        self.episodes = {}
        self.worker_ep = np.zeros(worker_size)
        self.workers = np.arange(worker_size)
        self.n_step = n_step - 1
        self.gamma = gamma
        
    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        nxt_idxs = (np.arange(self.worker_size+1) + self._next_idx) % self._maxsize
        self._next_idx = nxt_idxs[-1]; nxt_idxs = nxt_idxs[:-1]
        episode_keys = list(zip(self.workers,self.worker_ep[self.workers]))
        eplens = []
        for nidx,epkey in zip(nxt_idxs,episode_keys):
            if epkey not in self.episodes:
                self.episodes[epkey] = []
            self.episodes[epkey].append(nidx)
            eplens.append(len(self.episodes[epkey]))
            t = self._storage['terminals'][nidx]
            if t:
                del self.episodes[tuple(self._storage['episode'][nidx,:2])]
        self._storage['episode'][nxt_idxs] = np.concatenate(np.array(episode_keys,dtype=np.int32),np.expand_dims(np.array(eplens),axis=1),axis=1)
        self._add(nxt_idxs, obs_t, action, reward, nxtobs_t, done, terminal)
        for w,t in enumerate(terminal):
            if t:
                self.worker_ep[w] += 1
        if self._next_idx > self._len:
            self._len = self._next_idx
        else:
            self._len = self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, nxtobses_t, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, nxtobs_t, done, episode_key_and_idx, _ = data
            episode_key, episode_index = episode_key_and_idx
            nstep_idxs = self.episodes[episode_key][episode_index:(episode_index+self.n_step)]
            gamma = self.gamma
            for nidxes in nstep_idxs:                   #for nn,nidxes for enumerate(nstep_idxs)
                data = self._storage[nidxes]
                _, _, r, nxtobs_t, done, _, _ = data
                reward += gamma*r                       #for less computation then np.power(self.gamma,nn+1)*r 
                gamma *= self.gamma
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            nxtobses_t.append(nxtobs_t)
            dones.append(done)
        obses_t = [np.array(o) for o in list(zip(*obses_t))]
        actions = np.array(actions)
        rewards = np.array(rewards)
        nxtobses_t = [np.array(no) for no in list(zip(*nxtobses_t))]
        dones = np.array(dones)
        return (obses_t,
                actions,
                rewards,
                nxtobses_t,
                dones)
    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        nxt_idxs = []
        discounted_rewards = []
        for i in idxes:
            episode_key, episode_index = self._storage['episodes'][i]
            nstep_idxs = self.episodes[episode_key][episode_index:(episode_index+self.n_step)]
            gamma = self.gamma
            reward = self._storage['reward'][i]
            for nidxes in nstep_idxs:                   #for nn,nidxes for enumerate(nstep_idxs)
                reward += gamma*self._storage['reward'][nidxes]                       #for less computation then np.power(self.gamma,nn+1)*r 
                gamma *= self.gamma
            nxt_idxs.append(nstep_idxs[-1])
            discounted_rewards.append(reward)
        nxt_idxs = np.array(nxt_idxs)
        discounted_rewards = np.array(discounted_rewards)
        return {
            'obses'     : [self._storage[o][idxes] for o in self.obsdict.keys()],
            'actions'   : self._storage['actions'][idxes],
            'rewards'   : discounted_rewards,
            'nxtobses'  : [self._storage[no][nxt_idxs] for no in self.nextobsdict.keys()],
            'dones'     : self._storage['dones'][nxt_idxs],
            #'terminals'     : self._storage['terminals'][nxt_idxs]
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
        idxes = [random.randint(0, self._len - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
class PrioritizedEpisodicReplayBuffer(EpisodicReplayBuffer):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1, n_step=3, gamma=0.99, alpha = 0.4):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        super(PrioritizedEpisodicReplayBuffer, self).__init__(size, observation_space, worker_size, action_space, n_step, gamma)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, nxtobs_t, done, terminal):
        nxt_idxs = (np.arange(self.worker_size+1) + self._next_idx) % self._maxsize
        self._next_idx = nxt_idxs[-1]; nxt_idxs = nxt_idxs[:-1]
        episode_keys = list(zip(self.workers,self.worker_ep[self.workers]))
        eplens = []
        for nidx,epkey in zip(nxt_idxs,episode_keys):
            if epkey not in self.episodes:
                self.episodes[epkey] = []
            self.episodes[epkey].append(nidx)
            eplens.append(len(self.episodes[epkey]))
            t = self._storage['terminals'][nidx]
            if t:
                del self.episodes[tuple(self._storage['episode'][nidx,:2])]
        self._storage['episode'][nxt_idxs] = np.concatenate(np.array(episode_keys,dtype=np.int32),np.expand_dims(np.array(eplens),axis=1),axis=1)
        self._add(nxt_idxs, obs_t, action, reward, nxtobs_t, done, terminal)
        self._it_sum[nxt_idxs] = self._max_priority ** self._alpha
        self._it_min[nxt_idxs] = self._max_priority ** self._alpha
        for w,t in enumerate(terminal):
            if t:
                self.worker_ep[w] += 1
        if self._next_idx > self._len:
            self._len = self._next_idx
        else:
            self._len = self._maxsize
    
    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, self._len - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = np.array(self._it_sum.find_prefixsum_idx(mass))
        return idx
    
    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = np.array((p_sample * len(self._storage)) ** (-beta) / max_weight)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample['weights'] = weights
        encoded_sample['indexes'] = idxes
        return encoded_sample
    
    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))