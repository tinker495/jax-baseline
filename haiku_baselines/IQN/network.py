import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
    def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dualing=False):
        super(Model, self).__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dualing = dualing
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear
            
        self.pi_mtx = jax.lax.stop_gradient(jnp.expand_dims(jnp.pi* np.arange(0,128, dtype=np.float32), axis=(0,1))) # [ 1 x 1 x 128]
        
    def __call__(self,feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape
        quaitle_shape = tau.shape
        feature_tile = jnp.reshape(jnp.tile(jnp.expand_dims(feature,axis=1),(1,quaitle_shape[1],1)),(feature_shape[0]*quaitle_shape[1],feature_shape[1])) # [ (batch x tau ) x feature ]
        costau = jnp.reshape(jnp.cos(jnp.expand_dims(tau,axis=2)*self.pi_mtx),(feature_shape[0]*quaitle_shape[1],128))                      # [ (batch x tau ) x 128 ]
        quantile_embedding = hk.Sequential([self.layer(feature_shape[-1]),jax.nn.relu])(costau)                                             # [ (batch x tau ) x feature ]
        mul_embedding = feature_tile*quantile_embedding
        if not self.dualing:
            q_net = jnp.swapaxes(jnp.reshape(
                hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[1],self.action_size[0])),1,2)
            return q_net
        else:
            v = jnp.tile(
                jnp.swapaxes(
                jnp.reshape(
                hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] +
                [
                    self.layer(1)
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[1],1)),1,2),
                (1,self.action_size[0],1))
            a = jnp.swapaxes(jnp.reshape(
                hk.Sequential(
                [
                    jax.nn.relu if i%2 == 1 else self.layer(self.node) for i in range(2*self.hidden_n)
                ] + 
                [
                    self.layer(self.action_size[0])
                ]
                )(mul_embedding)
                ,(feature_shape[0],quaitle_shape[1],self.action_size[0])),1,2)
            q = jnp.concatenate([v,a],axis=2)
            return q