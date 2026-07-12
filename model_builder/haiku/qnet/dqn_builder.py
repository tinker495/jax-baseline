import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.layers import NoisyLinear
from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import print_param


class Model(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2, noisy=False, dueling=False):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.dueling = dueling
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            return hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                    )
                ]
            )(feature)
        v = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(feature)
        a = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(feature)
        return v + a - jnp.mean(a, axis=1, keepdims=True)


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        model = hk.transform(
            lambda x: Model(
                action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
            )(x)
        )
        preproc_fn = preproc.apply
        model_fn = model.apply
        if key is not None:
            key1, key2, key3 = jax.random.split(key, num=3)
            pre_param = preproc.init(
                key1,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            model_param = model.init(
                key2,
                preproc.apply(
                    pre_param,
                    key3,
                    [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                ),
            )
            params = hk.data_structures.merge(pre_param, model_param)
            if print_model:
                print("------------------build-haiku-model--------------------")
                print_param("preprocess", pre_param)
                print_param("model", model_param)
                print("-------------------------------------------------------")
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return _model_builder
