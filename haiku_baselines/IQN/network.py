import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
from einops import rearrange, reduce, repeat
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
        self.layer = partial(self.layer, w_init=hk.initializers.VarianceScaling(scale=2), b_init=hk.initializers.VarianceScaling(scale=2))
            
        self.pi_mtx = jax.lax.stop_gradient(
                        repeat(jnp.pi* np.arange(0,128, dtype=np.float32),'m -> o m',o=1)
                      ) # [ 1 x 128]
        
    def __call__(self,feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape                                                                                   #[ batch x feature]
        batch_size = feature_shape[0]                                                                                   #[ batch ]
        quaitle_shape = tau.shape                                                                                       #[ tau ]
        feature_tile = repeat(feature,'b f -> (b t) f',t=quaitle_shape[1])                                          #[ (batch x tau) x feature]
        
        costau = jnp.cos(
                    rearrange(
                    repeat(tau,'b t -> b t m',m=128),
                    'b t m -> (b t) m'
                    )*self.pi_mtx)                                                                                      #[ (batch x tau) x 128]
        quantile_embedding = hk.Sequential([self.layer(feature_shape[1]),jax.nn.relu])(costau)                       #[ (batch x tau) x feature ]

        mul_embedding = feature_tile*quantile_embedding                                                                 #[ (batch x tau) x feature ]
        
        if not self.dueling:
            q_net = rearrange(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,'(b t) a -> b a t',b=batch_size, t=quaitle_shape[1])                                                   #[ batch x action x tau ]
            return q_net
        else:
            v = repeat(
                rearrange(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(1)
                ]
                )(mul_embedding)
                ,'(b t) o -> b o t',b=batch_size, t=quaitle_shape[1])                                                   #[ batch x 1 x tau ]
                ,'b o t -> b a o t',a=self.action_size[0])                                                            #[ batch x action x tau ]
            a = rearrange(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,'(b t) a -> b a o t',b=batch_size, o=1, t=quaitle_shape[1])                                                   #[ batch x action x tau ]
            q = hk.Reshape((self.action_size[0],self.support_n*self.support_n))(v + a - jnp.max(a, axis=(1,2), keepdims=True))
            return q