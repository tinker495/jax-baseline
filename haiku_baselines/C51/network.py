import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
from haiku_baselines.common.layers import NoisyLinear


class Model(hk.Module):
	def __init__(self,action_size,node=256,hidden_n=2,noisy=False,dueling=False,categorial_bar_n=51):
		super(Model, self).__init__()
		self.action_size = action_size
		self.node = node
		self.hidden_n = hidden_n
		self.noisy = noisy
		self.dueling = dueling
		self.categorial_bar_n = categorial_bar_n
		if not noisy:
			self.layer = hk.Linear
		else:
			self.layer = NoisyLinear
		self.layer = partial(self.layer, w_init=hk.initializers.VarianceScaling(scale=2), b_init=hk.initializers.VarianceScaling(scale=2))
		
	def __call__(self,feature: jnp.ndarray) -> jnp.ndarray:
		if not self.dueling:
			q = hk.Sequential(
				[
					self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
				] + 
				[
					self.layer(self.action_size[0]*self.categorial_bar_n),
					hk.Reshape((self.action_size[0],self.categorial_bar_n))
				]
				)(feature)
			return jax.nn.softmax(q,axis=2)
		else:
			v = hk.Sequential(
				[
					self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
				] + 
				[
					self.layer(self.categorial_bar_n),
					hk.Reshape((1,self.categorial_bar_n))
				]
				)(feature)
			a = hk.Sequential(
				[
					self.layer(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n)
				] + 
				[
					self.layer(self.action_size[0]*self.categorial_bar_n),
					hk.Reshape((self.action_size[0],self.categorial_bar_n))
				]
				)(feature)
			q = v + (a - jnp.mean(a, axis=1, keepdims=True))
			return jax.nn.softmax(q,axis=2) 