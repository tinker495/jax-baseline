import numpy as np
import abc as ABC
import haiku as hk
import jax
import jax.numpy as jnp
from typing import List

class NoisyLinear(hk.Module):
    def __init__(self,state_size,cnn_mode="normal"):
        super(NoisyLinear, self).__init__()
        
    def __call__(self,states) -> jnp.ndarray:
        pass