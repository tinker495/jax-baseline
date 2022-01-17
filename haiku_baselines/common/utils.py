import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import List
 
@jax.jit
def hard_update(new_tensors, old_tensors, steps: int, update_period: int):
  update = (steps % update_period == 0)
  return jax.tree_multimap(
      lambda new, old: jax.lax.select(update, new, old), new_tensors, old_tensors)
        
@jax.jit
def soft_update(new_tensors, old_tensors, tau : float):
    return jax.tree_multimap(
      lambda new, old: tau * new + (1.0 - tau) * old,
      new_tensors, old_tensors)
  
def truncated_mixture(quantiles, out_support):
  quantiles = jnp.concatenate(quantiles,axis=-1)
  sorted = jnp.sort(quantiles,axis=1)
  return sorted[:,:out_support]
    
@jax.jit
def convert_states(obs : List):
  return [(o* 255.0).astype(np.uint8) if len(o.shape) >= 4 else o for o in obs]

@jax.jit
def convert_jax(obs : List):
  return [jax.device_get(o).astype(jnp.float32)*(1.0/255.0) if len(o.shape) >= 4 else jax.device_get(o) for o in obs]

@jax.jit
def discounted(rewards,gamma=0.99): #lfilter([1],[1,-gamma],x[::-1])[::-1]
  _gamma = 1
  out = 0
  for r in rewards:
    out += r*_gamma
    _gamma *= gamma
  return out

def discount_with_terminal(rewards, dones, terminals, next_values, gamma):
  def f(ret, info):
    reward, done, term, nextval = info
    ret = reward + gamma * (ret * (1. - term) + nextval * (1. - done) * term)
    return ret, ret
  ret = rewards[-1] + gamma * next_values[-1] * (1. - dones[-1])
  _, discounted = jax.lax.scan(f, ret, (rewards[:-1], dones[:-1], terminals[:-1], next_values[:-1]),reverse=True)
  return jnp.append(jnp.flip(discounted),ret)
  

'''
def discount_with_terminal(rewards, dones, terminals, next_values, gamma):
  ret = rewards[-1] + gamma * next_values[-1] * (1. - dones[-1])
  discounted = [ret]
  for reward, done, term, nextval in zip(rewards[-2::-1], dones[-2::-1], terminals[-2::-1], next_values[-2::-1]):
    ret = reward + gamma * (ret * (1. - term) + nextval * (1. - done) * term) # fixed off by one bug
    discounted.append(ret)
  return discounted[::-1]
'''

def get_gaes(rewards, dones, terminals, values, next_values, gamma, lamda):
  last_gae_lam = 0
  delta = rewards[-1] + gamma * next_values[-1] * (1. - dones[-1]) - values[-1]
  last_gae_lam = delta + gamma * lamda * (1. - dones[-1]) * last_gae_lam
  advs = [last_gae_lam]
  for reward, done, value, nextval, term in zip(rewards[-2::-1], dones[-2::-1], values[-2::-1], next_values[-2::-1], terminals[-2::-1]):
    delta = reward + gamma * (nextval * (1. - done)) - value
    last_gae_lam = delta + gamma * lamda * (1. - term) * last_gae_lam
    advs.append(last_gae_lam)
  advs = jnp.array(advs[::-1])
  target = advs + values
  return advs, target