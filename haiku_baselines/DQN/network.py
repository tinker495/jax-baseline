import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dueling=False):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dueling = dueling
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear
        
    def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(feature)
            return q_net
        else:
            v = hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(1)
                ]
                )(feature)
            a = hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(feature)
            return v + (a - jnp.max(a, axis=1, keepdims=True))  