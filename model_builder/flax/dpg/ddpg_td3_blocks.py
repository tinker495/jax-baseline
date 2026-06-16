"""Shared actor/critic blocks for the deterministic DDPG/TD3 builders.

ddpg and td3 use the byte-identical deterministic ``Actor`` and ``Critic``
architecture; they differ only in critic wiring (DDPG one critic, TD3 twin critics),
which stays in each builder's ``model_builder_maker``. Mirrors the
``simba_ddpg_td3_blocks`` / ``simbav2_ddpg_td3_blocks`` siblings.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        action = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3)),
                jax.nn.tanh,
            ]
        )(feature)
        return action


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_factorized_uniform(3))]
        )(concat)
        return q_net
