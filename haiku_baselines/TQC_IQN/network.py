import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
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
		self.layer = partial(self.layer, w_init=hk.initializers.VarianceScaling(scale=2), b_init=hk.initializers.VarianceScaling(scale=2))
		
	def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
			linear = hk.Sequential(
				[
					self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
				] + 
				[
					self.layer(self.action_size[0]*2)
				]
				)(feature)
			mu, log_std = jnp.split(linear, 2, axis=-1)
			return mu, LOG_STD_MEAN + LOG_STD_SCALE*jax.nn.tanh(log_std / LOG_STD_SCALE) #jnp.clip(log_std,LOG_STD_MIN,LOG_STD_MAX)

class Quantile_Embeding(hk.Module):
	def __init__(self,embedding_size=256):
		super(Quantile_Embeding, self).__init__()
		self.layer = hk.Linear
		self.embedding_size = embedding_size
		self.pi_mtx = jax.lax.stop_gradient(
						repeat(jnp.pi* np.arange(0,128, dtype=np.float32),'m -> o m',o=1)
					  )

	def __call__(self,feature: jnp.ndarray,actions: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
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
		return mul_embedding

class Critic(hk.Module):
	def __init__(self,node=256,hidden_n=2):
		super(Critic, self).__init__()
		self.node = node
		self.hidden_n = hidden_n - 1
		self.layer = hk.Linear

	def __call__(self,embedding: jnp.ndarray, batch_size, quaitle_size) -> jnp.ndarray:
		q_net = rearrange(hk.Sequential(
			[
				self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
			] + 
			[
				self.layer(1)
			]
			)(embedding)
			,'(b t) o -> b (t o)',b=batch_size, t=quaitle_size)
		return q_net