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
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0]),
                    jax.nn.tanh
                ]
                )(feature)
            return action
        
class Critic(hk.Module):
    def __init__(self,node=256,hidden_n=2,support_n=200):
        super(Critic, self).__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.support_n = support_n
        self.layer = hk.Linear
        
    def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature,actions],axis=1)
        q1_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ]
            )(concat)
        q1_mean = self.layer(1)(q1_net)
        q1_cumsum = jnp.cumsum(jax.nn.softplus(self.layer(self.support_n)(q1_net)),axis=1)
        q1 = q1_mean + q1_cumsum - jnp.mean(q1_cumsum,axis=1,keepdims=True)
        q2_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ]
            )(concat)
        q2_mean = self.layer(1)(q2_net)
        q2_cumsum = jnp.cumsum(jax.nn.softplus(self.layer(self.support_n)(q2_net)),axis=1)
        q2 = q2_mean + q2_cumsum - jnp.mean(q2_cumsum,axis=1,keepdims=True)
        return q1,q2
    '''
    def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature,actions],axis=1)
        q1_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(self.support_n)
            ]
            )(concat)
        q2_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(self.support_n)
            ]
            )(concat)
        return q1_net,q2_net
    '''