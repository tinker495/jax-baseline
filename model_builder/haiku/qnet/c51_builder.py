import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.haiku.layers import NoisyLinear, network_body
from model_builder.haiku.qnet.dqn_builder import make_qnet_builder
from model_builder.model_config import MLPConfig


class Model(hk.Module):
    def __init__(
        self,
        action_size,
        network: MLPConfig,
        noisy=False,
        dueling=False,
        categorial_bar_n=51,
    ):
        super().__init__()
        self.action_size = action_size
        self.network = network
        self.dueling = dueling
        self.categorial_bar_n = categorial_bar_n
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q = network_body(feature, self.network, self.layer)
            q = self.layer(
                self.action_size[0] * self.categorial_bar_n,
                w_init=hk.initializers.RandomUniform(-0.03, 0.03),
            )(q)
            q = hk.Reshape((self.action_size[0], self.categorial_bar_n))(q)
            return jax.nn.softmax(q, axis=2)
        v = network_body(feature, self.network, self.layer)
        v = self.layer(self.categorial_bar_n, w_init=hk.initializers.RandomUniform(-0.03, 0.03))(v)
        v = hk.Reshape((1, self.categorial_bar_n))(v)
        a = network_body(feature, self.network, self.layer)
        a = self.layer(
            self.action_size[0] * self.categorial_bar_n,
            w_init=hk.initializers.RandomUniform(-0.03, 0.03),
        )(a)
        a = hk.Reshape((self.action_size[0], self.categorial_bar_n))(a)
        q = v + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(q, axis=2)


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    return make_qnet_builder(
        observation_space,
        Model,
        action_space,
        dueling_model,
        param_noise,
        policy_kwargs,
        categorial_bar_n=categorial_bar_n,
    )
