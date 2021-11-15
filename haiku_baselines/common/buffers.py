import random
from typing import Optional, List, Union

import jax
import numpy as np
import jax.numpy as jnp

from haiku_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(object):
    def __init__(self, size: int, observation_space: list,worker_size = 1,action_space = 1, alpha = None, n_step=1, gamma=0.99):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._maxsize = size
        self.observation_space = observation_space
        self.action_space = action_space
        self._next_idx = 0
        self._len = 0
        self.n_step = n_step
        self.n_step_method = self.n_step > 1
        self.gamma = gamma
        self.alpha = alpha
        self.per = alpha is not None
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": jnp.uint8} if len(o) >= 3 else {"shape": o})
                            for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o,"dtype": jnp.uint8} if len(o) >= 3 else {"shape": o})
                            for idx,o in enumerate(observation_space))
        self._storage = dict(
            *(
            [(
                key, jnp.zeros((self._maxsize,*state["shape"]),dtype=state['dtype'] if 'dtype' in state else jnp.float32)
            )   for key, state in self.obsdict]
            +
            [(
                'rewards', jnp.zeros((self._maxsize,1),dtype=jnp.float32)
            )
            ,
            (
                'actions', jnp.zeros((self._maxsize,self.action_space),dtype=jnp.float32)
            )]
            +
            [(
                key, jnp.zeros((self._maxsize,*state["shape"]),dtype=state['dtype'] if 'dtype' in state else jnp.float32)
            )   for key,state in self.nextobsdict]
            +[(
                'dones', jnp.zeros((self._maxsize,1),dtype=jnp.float32)
            )]
            )
            )
        self.episodes = None
        self.worker_ep = None
        if self.n_step_method:
            self.episodes = {}
            self.worker_ep = jnp.zeros(worker_size)
            
        self._it_sum = None
        self._it_min = None
        self._max_priority = None
        if self.per:
            it_capacity = 1
            while it_capacity < size:
                it_capacity *= 2
            self._it_sum = SumSegmentTree(it_capacity)
            self._it_min = MinSegmentTree(it_capacity)
            self._max_priority = 1.0

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

    def add(self, obs_t, action, reward, nxtobs_t, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        obsdict = dict(zip(self.obsdict.keys(),obs_t))
        nextobsdict = dict(zip(self.nextobsdict.keys(),nxtobs_t))
        for k,data in obsdict:
            self._storage[k].at(self._next_idx).set(data)
        for k,data in nextobsdict:
            self._storage[k].at(self._next_idx).set(data)
        
        if self._next_idx >= self._len:
            self._len += 1
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, nxtobses_t, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, nxtobs_t, done = data
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
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, nxtobs_t, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, nxtobs_t, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
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
        return encoded_sample + (weights, idxes)

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

        self._max_priority = max(self._max_priority, np.max(priorities))*0.95
        
    
class EpisodicReplayBuffer(ReplayBuffer):
    def __init__(self, size, worker_size, n_step, gamma):
        """
        Create Episodic Replay buffer for n-step td

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(EpisodicReplayBuffer, self).__init__(size)
        self.episodes = {}
        self.worker_ep = np.zeros(worker_size)
        self.n_step = n_step - 1
        self.gamma = gamma
        
    def add(self, obs_t, action, reward, nxtobs_t, done, worker, terminal):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        episode_key = (worker,self.worker_ep[worker])
        if episode_key not in self.episodes:
            self.episodes[episode_key] = []
        self.episodes[episode_key].append(self._next_idx)
        data = (obs_t, action, reward, nxtobs_t, done, (episode_key,len(self.episodes[episode_key])), terminal)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            if self._storage[self._next_idx][6]: #remove episode data when remove last episode from storage
                del self.episodes[self._storage[self._next_idx][5][0]]
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if terminal:
            self.worker_ep[worker] += 1

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
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
class PrioritizedEpisodicReplayBuffer(EpisodicReplayBuffer):
    def __init__(self, size, worker_size, n_step, gamma, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedEpisodicReplayBuffer, self).__init__(size, worker_size, n_step, gamma)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, nxtobs_t, done, worker, terminal):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, nxtobs_t, done, worker, terminal)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
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
        return encoded_sample + (weights, idxes)

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

        self._max_priority = max(self._max_priority, np.max(priorities))*0.95