import numpy as np
import abc as ABC
import haiku as hk
import jax
import jax.numpy as jnp
from typing import List
    
class PreProcess(hk.Module):
    def __init__(self,state_size,cnn_mode="normal"):
        super(PreProcess, self).__init__()
        self.embedding = [
            visual_embedding(cnn_mode)
            if len(st) == 3 else lambda x: x
            for st in state_size 
        ]
        
    def __call__(self,states: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre,x in zip(self.embedding,states)],axis=1)
    
def visual_embedding(mode="simple"):
    if mode == "normal":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'), jax.nn.relu,
                    hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'), jax.nn.relu,
                    hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'), jax.nn.relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "simple":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(16, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'), jax.nn.relu,
                    hk.Conv2D(32, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'), jax.nn.relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "minimum":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(16, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'), jax.nn.relu,
                    hk.Flatten()
                    ])(x)
    elif mode == 'slide':
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(512, kernel_shape=[3, 3], stride=[1, 1], padding='SAME'), jax.nn.relu,
                    hk.Conv2D(512, kernel_shape=[3, 3], stride=[1, 1], padding='SAME'), jax.nn.relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "none":
        net_fn = hk.Flatten()
    return net_fn