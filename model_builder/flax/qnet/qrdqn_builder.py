from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, NoisyDense, network_body
from model_builder.flax.qnet.dqn_builder import make_qnet_builder
from model_builder.model_config import MLPConfig


class Model(nn.Module):
    action_size: Sequence[int]
    network: MLPConfig
    noisy: bool
    dueling: bool
    support_n: int

    def setup(self) -> None:
        if not self.noisy:
            self.layer = Dense
        else:
            self.layer = NoisyDense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = network_body(feature, self.network, self.layer)
            q_net = self.layer(
                self.action_size[0] * self.support_n,
                kernel_init=clip_factorized_uniform(3 / self.support_n),
            )(q_net)
            return jnp.reshape(q_net, (q_net.shape[0], self.action_size[0], self.support_n))
        v = network_body(feature, self.network, self.layer)
        v = self.layer(
            self.support_n,
            kernel_init=clip_factorized_uniform(3 / self.support_n),
        )(v)
        v = jnp.reshape(v, (v.shape[0], 1, self.support_n))
        a = network_body(feature, self.network, self.layer)
        a = self.layer(
            self.action_size[0] * self.support_n,
            kernel_init=clip_factorized_uniform(3 / self.support_n),
        )(a)
        a = jnp.reshape(a, (a.shape[0], self.action_size[0], self.support_n))
        q = v + a - jnp.mean(a, axis=1, keepdims=True)
        return q


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, support_n, policy_kwargs
):
    return make_qnet_builder(
        observation_space,
        Model,
        action_space,
        dueling_model,
        param_noise,
        policy_kwargs,
        support_n=support_n,
    )
