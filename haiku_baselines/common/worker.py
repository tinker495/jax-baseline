import os
import argparse
import gym

import jax
import jax.numpy as jnp
import numpy as np
import ray

class gymworker:
    
    def __init__(self,env_id):
        