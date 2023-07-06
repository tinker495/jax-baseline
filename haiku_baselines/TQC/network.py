import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial

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