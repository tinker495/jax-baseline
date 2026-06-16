"""Shared actor/critic blocks for the Haiku DPG builders.

ddpg and td3 share a byte-identical deterministic ``Actor`` (tanh head). sac and
tqc share a byte-identical gaussian ``Actor`` (mu / log_std head). ddpg, td3 and
sac share a byte-identical single-Q ``Critic`` (the twin-critic vs single-critic
wiring stays in each builder's ``model_builder_maker``; tqc keeps its own
``support_n`` critic and td7 keeps its embedding-aware actor/critic).

Haiku derives a module's param scope from its ``hk.Module`` subclass name
(lowercased), so each class is named so that its scope stays ``actor`` / ``critic``
to keep checkpoint keypaths byte-identical with the pre-extraction builders. The
gaussian actor passes ``name="actor"`` explicitly because its distinct class name
would otherwise produce a different scope. Mirrors the flax ``ddpg_td3_blocks``
sibling.
"""

import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.haiku.layers import LOG_STD_MEAN, LOG_STD_SCALE


class Actor(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)),
                jax.nn.tanh,
            ]
        )(feature)


class GaussianActor(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2):
        super().__init__(name="actor")
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        linear = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(
                    self.action_size[0] * 2, w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                )
            ]
        )(feature)
        mu, log_std = jnp.split(linear, 2, axis=-1)
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std / LOG_STD_SCALE)


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(concat)
