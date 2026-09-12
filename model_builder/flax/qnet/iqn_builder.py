from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, NoisyDense
from model_builder.flax.Module import PreProcess
from model_builder.model_config import ACTIVATIONS, MLPConfig
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

    def setup(self) -> None:
        if not self.noisy:
            self.layer = Dense
        else:
            self.layer = NoisyDense

        self.pi_mtx = jax.lax.stop_gradient(
            jnp.expand_dims(jnp.pi * (jnp.arange(0, 128, dtype=np.float32) + 1), axis=(0, 2))
        )  # [ 1 x 128 x 1]

    @nn.compact
    def __call__(self, feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape  # [ batch x feature]

        tau = jnp.expand_dims(tau, axis=1)  # [ batch x 1 x tau]
        costau = jnp.cos(tau * self.pi_mtx)  # [ batch x 128 x tau]

        def qnet(feature, costau):  # [ batch x feature], [ batch x 128 ]
            quantile_embedding = nn.Sequential([self.layer(feature_shape[1]), jax.nn.relu])(
                costau
            )  # [ batch x feature ]
            mul_embedding = feature * quantile_embedding  # [ batch x feature ]
            if not self.dueling:
                q_net = nn.Sequential(
                    [
                        layer
                        for config in self.network.layers
                        for layer in (self.layer(config.units), ACTIVATIONS[config.activation])
                    ]
                    + [self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))]
                )(mul_embedding)
                return q_net
            v = nn.Sequential(
                [
                    layer
                    for config in self.network.layers
                    for layer in (self.layer(config.units), ACTIVATIONS[config.activation])
                ]
                + [self.layer(1, kernel_init=clip_factorized_uniform(3))]
            )(mul_embedding)
            a = nn.Sequential(
                [
                    layer
                    for config in self.network.layers
                    for layer in (self.layer(config.units), ACTIVATIONS[config.activation])
                ]
                + [self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))]
            )(mul_embedding)
            q = v + a - jnp.mean(a, axis=1, keepdims=True)
            return q

        out = jax.vmap(qnet, in_axes=(None, 2), out_axes=2)(
            feature, costau
        )  # [ batch x action x tau ]
        return out


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
    policy_kwargs = qnet_model_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=policy_kwargs["network"].embedding_mode,
                    pre_postprocess=nn.Dense(512),
                )
                self.qnet = Model(
                    action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
                )

            def __call__(self, x, tau):
                x = self.preproc(x)
                return self.qnet(x, tau)

            def preprocess(self, x):
                return self.preproc(x)

            def q(self, x, tau):
                return self.qnet(x, tau)

        model = Merged()
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        model_fn = get_apply_fn_flax_module(model, model.q)
        if key is not None:
            tau = jax.random.uniform(key, (1, 2))
            observation = dummy_observation(observation_space)
            params = model.init(key, observation, tau)
            print_flax_model_summary(print_model, key, (model, observation, tau))
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return model_builder
