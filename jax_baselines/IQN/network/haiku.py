import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.common.utils import print_param
from jax_baselines.model.haiku.apply import get_apply_fn_haiku_module
from jax_baselines.model.haiku.layers import NoisyLinear
from jax_baselines.model.haiku.Module import PreProcess


class Model(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2, noisy=False, dueling=False):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
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

        tau = jnp.expand_dims(tau, axis=1)  # [ batch x 1 x tau]
        costau = jnp.cos(tau * self.pi_mtx)  # [ batch x 128 x tau]

        def qnet(feature, costau):  # [ batch x feature], [ batch x 128 ]
            quantile_embedding = hk.Sequential([self.layer(feature_shape[1]), jax.nn.relu])(
                costau
            )  # [ batch x feature ]

            mul_embedding = feature * quantile_embedding  # [ batch x feature ]
            if self.hidden_n != 0:
                mul_embedding = hk.Sequential(
                    [
                        self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                        for i in range(2 * self.hidden_n)
                    ]
                )(mul_embedding)
            if not self.dueling:
                q_net = self.layer(
                    self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                )(mul_embedding)
                return q_net
            else:
                v = self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))(mul_embedding)
                a = self.layer(
                    self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                )(mul_embedding)
                q = v + a - jnp.max(a, axis=(1), keepdims=True)
                return q

        out = jax.vmap(qnet, in_axes=(None, 2), out_axes=2)(
            feature, costau
        )  # [ batch x action x tau ]
        return out


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
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
            lambda x, tau: Model(
                action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
            )(x, tau)
        )
        preproc_fn = get_apply_fn_haiku_module(preproc)
        model_fn = get_apply_fn_haiku_module(model)
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            tau = jax.random.uniform(key4, (1, 64))
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
                tau,
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
