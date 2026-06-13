"""Shared Simba actor/critic blocks for the deterministic DDPG/TD3 builders.

simba_ddpg and simba_td3 use the byte-identical residual ``Actor`` and ``Critic``
architecture; they differ only in critic wiring (DDPG one critic, TD3 twin critics),
which stays in each builder's ``model_builder_maker``.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, ResidualBlock


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        action = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [
                nn.LayerNorm(),
                Dense(self.action_size[0], kernel_init=clip_factorized_uniform(3)),
                jax.nn.tanh,
            ]
        )(feature)
        return action


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [nn.LayerNorm(), Dense(1, kernel_init=clip_factorized_uniform(3))]
        )(concat)
        return q_net
