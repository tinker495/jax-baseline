import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dueling=False, embedding_size = 128):
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
        self.embedding_size = embedding_size
            
        self.pi_mtx = jax.lax.stop_gradient(jnp.expand_dims(jnp.pi* np.arange(0,128, dtype=np.float32), axis=0)) # [ 1 x 128]
        
    def __call__(self,feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape                                                                                   #[ batch x feature]
        quaitle_shape = tau.shape                                                                                       #[ tau ]
        feature_net = hk.Sequential(
                                    [
                                        self.layer(self.embedding_size),
                                        jax.nn.relu
                                    ]
                                    )(feature)
        feature_tile = jnp.tile(jnp.expand_dims(feature_net,axis=1),(1,quaitle_shape[0],1))                             #[ batch x tau x self.embedding_size]
        costau = jnp.cos(jnp.expand_dims(tau,axis=1)*self.pi_mtx)                                                       #[ tau x 128]
        quantile_embedding = jnp.expand_dims(
                             hk.Sequential([self.layer(self.embedding_size),jax.nn.relu])(costau),                      #[ tau x self.embedding_size ]
                             axis=0),                                                                                   #[ 1 x tau x self.embedding_size ]

        mul_embedding = jnp.reshape(feature_tile*quantile_embedding,
                        (feature_shape[0]*quaitle_shape[0],self.embedding_size))                                        #[ (batch x tau) x self.embedding_size ]
        if not self.dueling:
            q_net = jnp.swapaxes(jnp.reshape(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[0],self.action_size[0])),1,2)
            return q_net
        else:
            v = jnp.tile(
                jnp.swapaxes(
                jnp.reshape(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(1)
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[0],1)),1,2),
                (1,self.action_size[0],1))
            a = jnp.swapaxes(jnp.reshape(
                hk.Sequential(
                [
                    self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[0],self.action_size[0])),1,2)
            q = jnp.concatenate([v,a],axis=2)
            return q