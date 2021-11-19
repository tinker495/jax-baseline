import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dualing=False,categorial_bar_n=51):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dualing = dualing
        self.categorial_bar_n = categorial_bar_n
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
                    self.layer(self.action_size[0]*self.categorial_bar_n)
                ]
                )(feature)
            q = jnp.reshape(q_net,(-1,self.action_size[0],self.categorial_bar_n))
            return jax.nn.softmax(q,axis=2)
        else:
            q_net = hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ]
                )(feature)
            v = jnp.reshape(self.layer(self.categorial_bar_n)(q_net),(-1,1,self.categorial_bar_n))
            a = jnp.reshape(self.layer(self.action_size[0]*self.categorial_bar_n)(q_net),(-1,self.action_size[0],self.categorial_bar_n))
            q = v + (a - jnp.mean(a, axis=1, keepdims=True))
            return jax.nn.softmax(q,axis=2) 