import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dualing=False,support_n=200):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dualing = dualing
        self.support_n = support_n
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
                    self.layer(self.action_size[0]*self.support_n),
                    hk.Reshape((self.action_size[0],self.support_n))
                ]
                )(feature)
            return q_net
        else:
            v = jnp.tile(
                hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(self.support_n),
                    hk.Reshape((1,self.support_n))
                ]
                )(feature),
                (1,self.action_size[0],1))
            a = hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(self.action_size[0]*self.support_n),
                    hk.Reshape((self.action_size[0],self.support_n))
                ]
                )(feature)
            q = jnp.concatenate([v,a],axis=2)
            return q