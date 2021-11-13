import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dualing=False):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dualing = dualing
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear
        
    def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dualing:
            q_net = hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(feature)
            return q_net
        else:
            q_net = hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ]
                )(feature)
            v = self.layer(1)(q_net)
            a = self.layer(self.action_size[0])(q_net)
            return v + (a - jnp.mean(a, axis=1, keepdims=True))  