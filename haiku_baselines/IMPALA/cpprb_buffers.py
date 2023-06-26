import random
from typing import Dict, Optional, List, Union
from collections import deque, namedtuple

import multiprocessing as mp
import numpy as np
import cpprb
from ray.util.queue import Queue
import copy

batch = namedtuple('batch_tuple', ['obses', 'actions', 'mu_log_prob', 'rewards','nxtobses', 'dones', 'terminals'])

class EpochBuffer:
    def __init__(self, size: int, env_dict:dict):
        self.max_size = size
        self.env_dict = env_dict
        self.obsdict = dict((o,s) for o,s in env_dict.items() if o.startswith("obs"))
        self.nextobsdict = dict((o,s) for o,s in env_dict.items() if o.startswith("next_obs"))
        self.buffer = cpprb.ReplayBuffer(size, env_dict = env_dict)

    def __len__(self):
        return self.buffer.get_stored_size()
    
    def add(self, obs_t, action, log_prob, reward, nxtobs_t, done, terminal):
        obsdict = dict(zip(self.obsdict.keys(),[o for o in obs_t]))
        nextobsdict = dict(zip(self.nextobsdict.keys(),[no for no in nxtobs_t]))
        self.buffer.add(**obsdict,action=action,log_prob=log_prob,reward=reward,**nextobsdict,done=done,terminal=terminal)
        if done or terminal:
            self.buffer.on_episode_end()

    def get_buffer(self):
        trans = self.buffer.get_all_transitions()
        transitions = batch([trans[o] for o in self.obsdict.keys()],
                            trans['action'],
                            trans['log_prob'],
                            trans['reward'],
                            [trans[o] for o in self.nextobsdict.keys()],
                            trans['done'],
                            trans['terminal'])
        return transitions

class ImpalaBuffer:
    def __init__(self, size: int, actor_num : int, observation_space: list, discrete=True, action_space = 1, manager = None, compress_memory = False):
        self.max_size = size
        self.actor_num = actor_num
        self.replay = size > 0
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                        for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("next_obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                        for idx,o in enumerate(observation_space))
        self.obscompress = None
        if compress_memory:
            self.obscompress = []
            for k in self.obsdict:
                if len(self.obsdict[k]["shape"]) >= 3:
                    self.obscompress.append(k)
                    del self.nextobsdict[f'next_{k}']

        self.env_dict = {**self.obsdict,
                    "action": {"shape": 1 if discrete else action_space},
                    "log_prob": {},
                    "reward": {},
                    **self.nextobsdict,
                    "done": {},
                    "terminal": {}
                    }
        
        self.queue = Queue(maxsize=actor_num*2)
        if self.replay:
            self.replay_buffer = deque(maxlen=size)
            self.sample = self.replay_sample
        else:
            self.sample = self.queue_sample

    def queue_info(self):
        return self.queue, self.env_dict, self.actor_num
        
    def __len__(self):
        return len(self.queue)
    
    def queue_is_empty(self):
        return self.queue.empty()
    
    def queue_sample(self, stack_size: int):
        gets = [self.queue.get() for idx in range(stack_size)]
        transitions = batch([get[0] for get in gets],
                            [get[1] for get in gets],
                            [get[2] for get in gets],
                            [get[3] for get in gets],
                            [get[4] for get in gets],
                            [get[5] for get in gets],
                            [get[6] for get in gets])
        return transitions

    def replay_sample(self, stack_size: int):
        while len(self.replay_buffer) < stack_size or not self.queue.empty():
            self.replay_buffer.append(self.queue.get())
        gets = random.sample(self.replay_buffer, stack_size)
        transitions = batch([get[0] for get in gets],
                            [get[1] for get in gets],
                            [get[2] for get in gets],
                            [get[3] for get in gets],
                            [get[4] for get in gets],
                            [get[5] for get in gets],
                            [get[6] for get in gets])
        return transitions