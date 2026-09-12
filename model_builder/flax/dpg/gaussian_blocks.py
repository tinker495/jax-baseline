"""Configured squashed-Gaussian actor shared by SAC and TQC."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import (
    LOG_STD_MEAN,
    LOG_STD_SCALE,
    Dense,
    SimbaV2Head,
    network_body,
)
from model_builder.model_config import DEFAULT_MLP, ModelConfig, ResidualConfig


class Actor(nn.Module):
    action_size: tuple
    network: ModelConfig = DEFAULT_MLP
    layer: type[nn.Module] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        features = network_body(features, self.network, self.layer)
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            mu = SimbaV2Head(self.network.blocks[-1], self.action_size[0])(features)
            log_std = SimbaV2Head(self.network.blocks[-1], self.action_size[0])(features)
        else:
            mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))(features)
            log_std = self.layer(
                self.action_size[0],
                kernel_init=clip_factorized_uniform(3),
                bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
            )(
                features
            )  # initialize std with high values
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std / LOG_STD_SCALE)
