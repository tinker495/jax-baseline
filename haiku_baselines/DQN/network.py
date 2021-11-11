import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dualing=False):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dualing = dualing
        
    def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dualing:
            q_net = hk.Sequential(
                [
                    jax.nn.relu if i%2 else hk.Linear(self.node) for i in range(2*self.hidden_n)
                ] + 
                [
                    hk.Linear(self.action_size[0])
                ]
                )(feature)
            return q_net
        else:
            q_net = hk.Sequential(
                [
                    jax.nn.relu if i%2 else hk.Linear(self.node) for i in range(2*self.hidden_n)
                ]
                )(feature)
            v = hk.Linear(1)(q_net)
            a = hk.Linear(self.action_size[0])(q_net)
            return v + (a - jnp.mean(a, axis=1))  