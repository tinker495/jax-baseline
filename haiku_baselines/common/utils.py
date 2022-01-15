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

def get_gaes(rewards, dones, terminals, values, next_values, gamma, lamda, normalize):
    deltas = rewards + gamma * (1.0 - dones) * next_values - values
    
    gaes = jnp.array(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes.at[t].set(gaes[t] + (1.0 - terminals[t]) * gamma * lamda * gaes[t + 1])

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target