"""Shared actor/critic blocks for the stochastic Gaussian DPG builders.

sac, dac and tqc share the byte-identical squashed-Gaussian ``Actor``
(``(mu, log_std)`` head). sac and dac additionally share the byte-identical
plain ``Critic`` (single ``Dense(1)`` head); tqc keeps its own quantile critic
and dac keeps its own ``Optimistic_Actor``. Per-builder critic wiring (twin
critics, optimistic actor) stays in each builder's ``model_builder_maker``.
Mirrors the deterministic ``ddpg_td3_blocks`` sibling.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import LOG_STD_MEAN, LOG_STD_SCALE, Dense


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        linear = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))(linear)
        log_std = self.layer(
            self.action_size[0],
            kernel_init=clip_factorized_uniform(3),
            bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
        )(
            linear
        )  # initialize std with high values
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std / LOG_STD_SCALE)


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
