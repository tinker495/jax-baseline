import numpy as np
import abc as ABC
import haiku as hk
import jax
import jax.numpy as jnp
from typing import List
    
class PreProcess(hk.Module):
    def __init__(self,state_size,cnn_mode="normal"):
        super(PreProcess, self).__init__()
        self.embedding = [
            visual_embedding(cnn_mode)
            if len(st) == 3 else lambda x: x
            for st in state_size 
        ]
        
    def __call__(self,states: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre,x in zip(self.embedding,states)],axis=1)
    
class FractionProposal(hk.Module):
    def __init__(self,support_size,node=256,hidden_n=2):
        self.support_size = support_size
        self.node = node
        self.hidden_n = hidden_n
        
    def __call__(self,feature):
        batch = feature.shape[0]
        log_probs = hk.Sequential([
                    hk.Linear(self.node) if i%2 == 0 else jax.nn.relu for i in range(2*self.hidden_n - 1)
                    ]+[jax.nn.log_softmax]
                    )
        probs = jnp.exp(log_probs)
        tau_0 = jnp.zeros((batch,1),dtype=np.float32)
        tau_1_N = jnp.cumsum(probs,axis=1)
        tau = jnp.concatenate([tau_0,tau_1_N],axis=1)
        tau_hat = jax.lax.stop_gradient((tau[:,:-1] + tau[:,1:])/2.0)
        entropy = -jnp.sum(log_probs * probs,keepdims=True)
        return tau, tau_hat, entropy
    
def visual_embedding(mode="simple"):
    if mode == "normal":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'), jax.nn.relu6,
                    hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'), jax.nn.relu6,
                    hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'), jax.nn.relu6,
                    hk.Flatten()
                    ])(x)
    elif mode == "simple":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(16, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'), jax.nn.relu6,
                    hk.Conv2D(32, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'), jax.nn.relu6,
                    hk.Flatten()
                    ])(x)
    elif mode == "minimum":
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(16, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'), jax.nn.relu6,
                    hk.Flatten()
                    ])(x)
    elif mode == 'slide':
        net_fn = lambda x: hk.Sequential([
                    hk.Conv2D(512, kernel_shape=[3, 3], stride=[1, 1], padding='SAME'), jax.nn.relu6,
                    hk.Conv2D(512, kernel_shape=[3, 3], stride=[1, 1], padding='SAME'), jax.nn.relu6,
                    hk.Flatten()
                    ])(x)
    elif mode == "none":
        net_fn = hk.Flatten()
    return net_fn