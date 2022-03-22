import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp


class Actor(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2):
        super(Actor, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear
        
    def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
            action = hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu6 for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0]),
                    jax.nn.tanh
                ]
                )(feature)
            return action
        
class Critic(hk.Module):
    def __init__(self,node=256,hidden_n=2):
        super(Critic, self).__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear
        
    def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature,actions],axis=1)
        q_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu6 for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(1)
            ]
            )(concat)
        return q_net