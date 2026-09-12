import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.haiku.layers import NoisyLinear, network_body
from model_builder.haiku.Module import PreProcess
from model_builder.model_config import MLPConfig
from model_builder.utils import (
    dummy_observation,
    print_haiku_model_summary,
    qnet_model_kwargs,
)


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
    policy_kwargs = qnet_model_kwargs(policy_kwargs, allowed_embeddings=("normal",))

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(
                observation_space, embedding_mode=policy_kwargs["network"].embedding_mode
            )(x)
        )
        model = hk.transform(
            lambda x: Model(
                action_space,
                dueling=dueling_model,
                noisy=param_noise,
                categorial_bar_n=categorial_bar_n,
                **policy_kwargs,
            )(x)
        )
        preproc_fn = preproc.apply
        model_fn = model.apply
        if key is not None:
            key1, key2, key3 = jax.random.split(key, num=3)
            observation = dummy_observation(observation_space)
            pre_param = preproc.init(key1, observation)
            feature = preproc.apply(pre_param, key3, observation)
            model_param = model.init(key2, feature)
            params = hk.data_structures.merge(pre_param, model_param)
            print_haiku_model_summary(print_model, (preproc, observation), (model, feature))
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return _model_builder
