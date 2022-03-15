import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat


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
        self.embedding_size = node
            
        self.pi_mtx = jax.lax.stop_gradient(
                        repeat(jnp.pi* np.arange(0,128, dtype=np.float32),'m -> o m',o=1)
                      ) # [ 1 x 128]

    def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape                                                                                   #[ batch x feature]
        batch_size = feature_shape[0]                                                                                   #[ batch ]
        quaitle_shape = tau.shape                                                                                       #[ tau ]
        concat = jnp.concatenate([feature,actions],axis=1)
        feature_net = hk.Sequential(
                                    [
                                        self.layer(self.embedding_size),
                                        jax.nn.relu
                                    ]
                                    )(concat)
        feature_tile = repeat(feature_net,'b f -> (b t) f',t=quaitle_shape[1])                                          #[ (batch x tau) x self.embedding_size]
        
        costau = jnp.cos(
                    rearrange(
                    repeat(tau,'b t -> b t m',m=128),
                    'b t m -> (b t) m'
                    )*self.pi_mtx)                                                                                      #[ (batch x tau) x 128]
        quantile_embedding = hk.Sequential([self.layer(self.embedding_size),jax.nn.relu])(costau)                       #[ (batch x tau) x self.embedding_size ]

        mul_embedding = feature_tile*quantile_embedding                                                                 #[ (batch x tau) x self.embedding_size ]

        q1_net = rearrange(hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(1)
            ]
            )(mul_embedding)
            ,'(b t) o -> b (t o)',b=batch_size, t=quaitle_shape[1])
        q2_net = rearrange(hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(1)
            ]
            )(mul_embedding)
            ,'(b t) o -> b (t o)',b=batch_size, t=quaitle_shape[1])
        return q1_net,q2_net