import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from jax_baselines.model.flax.preprocesser import PreProcesser
from jax_baselines.model.flax.mlp import MLP


class Qnet(nn.Module):
    @flax.struct.dataclass
    class HyperParams:
        states_size: List[Tuple[int, ...]]
        action_size: int
        qnet_type: str = "mlp"
        embedding_mode: str = "normal"
        qnet: MLP.HyperParams = MLP.HyperParams()

    hyperparams: HyperParams

    preprocesser: Callable[[jnp.ndarray], jnp.ndarray] = None
    qnet_model: nn.Module = None

    def setup(self):
        if self.preprocesser is None:
            self.preprocesser = PreProcesser(embedding_mode=self.hyperparams.embedding_mode)
        if self.qnet_model is None:
            self.qnet_model = MLP(self.hyperparams.qnet)

    @nn.compact
    def __call__(self, states: List[jnp.ndarray]) -> jnp.ndarray:
        x = self.preprocesser(states)
        return self.qnet_model(x)
