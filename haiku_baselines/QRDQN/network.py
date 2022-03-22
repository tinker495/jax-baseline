import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dueling=False,support_n=200):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dueling = dueling
        self.support_n = support_n
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
                    self.layer(self.action_size[0]*self.support_n),
                    hk.Reshape((self.action_size[0],self.support_n))
                ]
                )(feature)
            return q_net
        else:
            v = repeat(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(self.support_n),
                    hk.Reshape((1,self.support_n))
                ]
                )(feature),
                'b o t -> b (a o) t',a = self.action_size[0])
            a = hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(self.action_size[0]*self.support_n),
                    hk.Reshape((self.action_size[0],self.support_n))
                ]
                )(feature)
            q = v + a - jnp.max(a,axis=1,keepdims=True)
            return q