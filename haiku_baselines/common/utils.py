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
    
@jax.jit
def convert_states(obs : List):
  return [(o* 256).astype(np.uint8) if len(o.shape) >= 4 else o for o in obs]

@jax.jit
def convert_jax(obs : List):
  return [jax.device_get(o).astype(jnp.float32) if len(o.shape) >= 4 else jax.device_get(o) for o in obs]