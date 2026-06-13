"""Shared SimbaV2 actor/critic blocks for the deterministic DDPG/TD3 builders.

simbav2_ddpg and simbav2_td3 use the byte-identical SimbaV2 ``Actor`` and ``Critic``
architecture; they differ only in critic wiring (DDPG one critic, TD3 twin critics),
which stays in each builder's ``model_builder_maker``.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.layers import SimbaV2Block, SimbaV2Embedding, SimbaV2Head


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoded = SimbaV2Embedding(self.node)(feature)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        logits = SimbaV2Head(self.node, self.action_size[0])(encoded)
        return jax.nn.tanh(logits)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        encoded = SimbaV2Embedding(self.node)(concat)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        q_value = SimbaV2Head(self.node, 1)(encoded)
        return q_value
