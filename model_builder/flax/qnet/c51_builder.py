from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, NoisyDense, network_body
from model_builder.flax.Module import PreProcess
from model_builder.model_config import MLPConfig
from model_builder.utils import (
    dummy_observation,
    print_flax_model_summary,
    qnet_model_kwargs,
)


class Model(nn.Module):
    action_size: Sequence[int]
    network: MLPConfig
    noisy: bool
    dueling: bool
    categorial_bar_n: int

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
                self.action_size[0] * self.categorial_bar_n,
                kernel_init=clip_factorized_uniform(0.01),
            )(q_net)
            q_net = jnp.reshape(q_net, (q_net.shape[0], self.action_size[0], self.categorial_bar_n))
            return jax.nn.softmax(q_net, axis=2)
        v = network_body(feature, self.network, self.layer)
        v = self.layer(self.categorial_bar_n, kernel_init=clip_factorized_uniform(0.01))(v)
        v = jnp.reshape(v, (v.shape[0], 1, self.categorial_bar_n))
        a = network_body(feature, self.network, self.layer)
        a = self.layer(
            self.action_size[0] * self.categorial_bar_n,
            kernel_init=clip_factorized_uniform(0.01),
        )(a)
        a = jnp.reshape(a, (a.shape[0], self.action_size[0], self.categorial_bar_n))
        q = v + a - jnp.mean(a, axis=1, keepdims=True)
        return jax.nn.softmax(q, axis=2)


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    policy_kwargs = qnet_model_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=policy_kwargs["network"].embedding_mode
                )
                self.qnet = Model(
                    action_space,
                    dueling=dueling_model,
                    noisy=param_noise,
                    categorial_bar_n=categorial_bar_n,
                    **policy_kwargs,
                )

            def __call__(self, x):
                x = self.preproc(x)
                return self.qnet(x)

            def preprocess(self, x):
                return self.preproc(x)

            def q(self, x):
                return self.qnet(x)

        model = Merged()
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        model_fn = get_apply_fn_flax_module(model, model.q)
        if key is not None:
            observation = dummy_observation(observation_space)
            params = model.init(key, observation)
            print_flax_model_summary(print_model, key, (model, observation))
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return model_builder
