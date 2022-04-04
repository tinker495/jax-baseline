import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce, repeat

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN)/2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN)/2.0

class Actor(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2):
        super(Actor, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

        self.pi_mtx = jax.lax.stop_gradient(
                        repeat(jnp.pi* np.arange(0,128, dtype=np.float32),'m -> o (a m)',o=1,a=self.action_size[0])
                      ) # [ 1 x 128]
        
    def __call__(self,feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape                                                                                   #[ batch x feature]
        batch_size = feature_shape[0]                                                                                   #[ batch ]
        quaitle_shape = tau.shape                                                                                       #[ batch x tau x actions]
        x = hk.Sequential([self.layer(self.node),jax.nn.relu])(feature)
        feature_tile = repeat(x,'b f -> (b t) f',t=quaitle_shape[1])                                              #[ (batch x tau) x feature]

        costau = jnp.cos(
                    rearrange(
                    repeat(tau,'b t a-> b t (a m)',m=128),
                    'b t am -> (b t) am'
                    )*self.pi_mtx)                                                                                      #[ (batch x tau) x (a x 128)]
        quantile_embedding = hk.Sequential([self.layer(self.node),jax.nn.relu])(costau)                       #[ (batch x tau) x feature ]

        mul_embedding = feature_tile*quantile_embedding                                                                 #[ (batch x tau) x feature ]

        actions = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(self.action_size[0])
            ]
            )(mul_embedding)                                                                                            #[ (batch x tau) x actions ]
            
        return rearrange(actions,'(b t) a -> b t a',b=batch_size, t=quaitle_shape[1])

class Critic(hk.Module):
    def __init__(self,node=256,hidden_n=2,support_n=200):
        super(Critic, self).__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.support_n = support_n
        self.layer = hk.Linear
        
    def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature,actions],axis=1)
        q_net = hk.Sequential(
            [
                self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
            ] + 
            [
                self.layer(self.support_n)
            ]
            )(concat)
        return q_net