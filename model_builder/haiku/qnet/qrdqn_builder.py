import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.apply import get_apply_fn_haiku_module
from model_builder.haiku.layers import NoisyLinear
from model_builder.haiku.Module import PreProcess
from model_builder.utils import print_param


class Model(hk.Module):
    def __init__(
        self, action_size, node=256, hidden_n=2, noisy=False, dueling=False, support_n=200
    ):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dueling = dueling
        self.support_n = support_n
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.support_n,
                        w_init=hk.initializers.RandomUniform(-0.03, 0.03),
                    ),
                    hk.Reshape((self.action_size[0], self.support_n)),
                ]
            )(feature)
            return q_net
        else:
            v = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(self.support_n, w_init=hk.initializers.RandomUniform(-0.03, 0.03)),
                    hk.Reshape((1, self.support_n)),
                ]
            )(feature)
            a = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.support_n,
                        w_init=hk.initializers.RandomUniform(-0.03, 0.03),
                    ),
                    hk.Reshape((self.action_size[0], self.support_n)),
                ]
            )(feature)
            q = v + a - jnp.max(a, axis=1, keepdims=True)
            return q


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, support_n, policy_kwargs
):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        model = hk.transform(
            lambda x: Model(
                action_space,
                dueling=dueling_model,
                noisy=param_noise,
                support_n=support_n,
                **policy_kwargs
            )(x)
        )
        preproc_fn = get_apply_fn_haiku_module(preproc)
        model_fn = get_apply_fn_haiku_module(model)
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
        else:
            return preproc_fn, model_fn

    return _model_builder
