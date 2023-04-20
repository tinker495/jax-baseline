import random
from typing import Optional, List, Union

import numpy as np
import cpprb

class IMPALA_Buffer(object):

    def __init__(self, epoch_size : int, observation_space: list, action_space = 1):
        self.epoch_size = epoch_size
        self.obsdict = dict(("obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("next_obs{}".format(idx),{"shape": o,"dtype": np.uint8} if len(o) >= 3 else {"shape": o,"dtype": np.float32})
                            for idx,o in enumerate(observation_space))
        self.buffer = cpprb.ReplayBuffer(epoch_size,
                        env_dict={**self.obsdict,
                            "action": {"shape": action_space},
                            "reward": {},
                            **self.nextobsdict,
                            "done": {},
                            "terminal": {}
                        })
        
    def add(self, obs, action, reward, next_obs, done, terminal):
        self.buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, terminal=terminal)


    def get_transitions(self):
        return self.buffer.get_all_transitions()

        
