import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.layers import NoisyLinear, network_body
from model_builder.haiku.Module import PreProcess
from model_builder.model_config import MLPConfig
from model_builder.utils import (
    dummy_observation,
    print_haiku_model_summary,
    qnet_model_kwargs,
)


class Model(hk.Module):
    def __init__(self, action_size, network: MLPConfig, noisy=False, dueling=False):
        super().__init__()
        self.action_size = action_size
        self.network = network
        self.dueling = dueling
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear

        self.pi_mtx = jax.lax.stop_gradient(
            jnp.expand_dims(jnp.pi * (jnp.arange(0, 128, dtype=np.float32) + 1), axis=(0, 2))
        )  # [ 1 x 128 x 1]

    def __call__(self, feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape  # [ batch x feature]
        tau_shape = tau.shape[-1]

        tau = jnp.expand_dims(tau, axis=1)  # [ batch x 1 x tau]
        costau = jnp.cos(tau * self.pi_mtx)  # [ batch x 128 x tau]
        chex.assert_shape(costau, (None, 128, tau_shape))

        def qnet(feature, costau):  # [ batch x feature], [ batch x 128 ]
            quantile_embedding = hk.Sequential([self.layer(feature_shape[1]), jax.nn.relu])(
                costau
            )  # [ batch x feature ]

            mul_embedding = feature * quantile_embedding  # [ batch x feature ]
            if not self.dueling:
                q = network_body(mul_embedding, self.network, self.layer)
                return self.layer(
                    self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                )(q)
            v = network_body(mul_embedding, self.network, self.layer)
            v = self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))(v)
            a = network_body(mul_embedding, self.network, self.layer)
            a = self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03))(
                a
            )
            return v + a - jnp.mean(a, axis=1, keepdims=True)

        out = jax.vmap(qnet, in_axes=(None, 2), out_axes=2)(
            feature, costau
        )  # [ batch x action x tau ]
        chex.assert_shape(out, (None, self.action_size[0], tau_shape))
        return out


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
    policy_kwargs = qnet_model_kwargs(policy_kwargs, allowed_embeddings=("normal",))

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(
                observation_space, embedding_mode=policy_kwargs["network"].embedding_mode
            )(x)
        )
        model = hk.transform(
            lambda x, tau: Model(
                action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
            )(x, tau)
        )
        preproc_fn = preproc.apply
        model_fn = model.apply
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            tau = jax.random.uniform(key4, (1, 64))
            observation = dummy_observation(observation_space)
            pre_param = preproc.init(key1, observation)
            feature = preproc.apply(pre_param, key3, observation)
            model_param = model.init(key2, feature, tau)
            params = hk.data_structures.merge(pre_param, model_param)
            print_haiku_model_summary(print_model, (preproc, observation), (model, feature, tau))
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return _model_builder
