"""Configured deterministic actor and scalar critic shared by DDPG, TD3 and SAC."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, SimbaV2Head, network_body
from model_builder.model_config import DEFAULT_MLP, ModelConfig, ResidualConfig


class Actor(nn.Module):
    action_size: tuple
    network: ModelConfig = DEFAULT_MLP
    layer: type[nn.Module] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        features = network_body(features, self.network, self.layer)
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            action = SimbaV2Head(self.network.width, self.action_size[0])(features)
        else:
            action = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))(
                features
            )
        return jax.nn.tanh(action)


class Critic(nn.Module):
    network: ModelConfig = DEFAULT_MLP
    layer: type[nn.Module] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        features = network_body(
            jnp.concatenate([features, actions], axis=1), self.network, self.layer
        )
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            return SimbaV2Head(self.network.width, 1)(features)
        return self.layer(1, kernel_init=clip_factorized_uniform(3))(features)
