import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from typing import Any, Callable, Dict, Optional, Tuple, Union, List


def visual_embedding(mode: str = "normal") -> Callable[[jnp.ndarray], jnp.ndarray]:
    if mode == "normal":
        net = nn.Sequential(
            nn.Conv(32, kernel_size=[8, 8], strides=[4, 4], padding="VALID"),
            nn.relu,
            nn.Conv(64, kernel_size=[4, 4], strides=[2, 2], padding="VALID"),
            nn.relu,
            nn.Conv(64, kernel_size=[3, 3], strides=[1, 1], padding="VALID"),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(512),
        )
    elif mode == "minimum":
        net = nn.Sequential(
            nn.Conv(16, kernel_size=[3, 3], strides=[1, 1], padding="VALID"),
            nn.relu,
            nn.Conv(32, kernel_size=[4, 4], strides=[2, 2], padding="VALID"),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(256),
        )
    elif mode == "none":
        net = lambda x: x.reshape((x.shape[0], -1))
    return net


class PreProcesser(nn.Module):
    states_size: List[Tuple[int, ...]]
    embedding_mode: str = "normal"
    embedding: List[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        if self.embedding is None:
            self.embedding = [
                visual_embedding(self.embedding_mode) if len(st) == 3 else lambda x: x
                for st in self.states_size
            ]

    @nn.compact
    def __call__(self, states: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre, x in zip(self.embedding, states)], axis=1)
