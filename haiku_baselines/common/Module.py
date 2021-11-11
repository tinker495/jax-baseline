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
        def net_fn(x) -> jnp.ndarray:
            return hk.Sequential([
                    hk.Conv2D(32,8,4), jax.nn.leaky_relu,
                    hk.Conv2D(64,4,2), jax.nn.leaky_relu,
                    hk.Conv2D(64,4,2), jax.nn.leaky_relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "simple":
        def net_fn(x) -> jnp.ndarray:
            return hk.Sequential([
                    hk.Conv2D(16,8,4), jax.nn.leaky_relu,
                    hk.Conv2D(32,4,2), jax.nn.leaky_relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "minimum":
        def net_fn(x) -> jnp.ndarray:
            return hk.Sequential([
                    hk.Conv2D(35,3,1), jax.nn.leaky_relu,
                    hk.Conv2D(144,3,1), jax.nn.leaky_relu,
                    hk.Flatten()
                    ])(x)
    elif mode == "none":
        net_fn = hk.Flatten()
    return net_fn