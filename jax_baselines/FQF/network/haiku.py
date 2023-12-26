import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

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
            repeat(jnp.pi * np.arange(0, 128, dtype=np.float32), "m -> o m", o=1)
        )  # [ 1 x 128]

    def __call__(self, feature: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        feature_shape = feature.shape  # [ batch x feature]
        batch_size = feature_shape[0]  # [ batch ]
        quaitle_shape = tau.shape  # [ tau ]
        feature_tile = repeat(
            feature, "b f -> (b t) f", t=quaitle_shape[1]
        )  # [ (batch x tau) x feature]

        costau = jnp.cos(
            rearrange(repeat(tau, "b t -> b t m", m=128), "b t m -> (b t) m") * self.pi_mtx
        )  # [ (batch x tau) x 128]
        quantile_embedding = hk.Sequential([self.layer(feature_shape[1]), jax.nn.relu])(
            costau
        )  # [ (batch x tau) x feature ]

        mul_embedding = feature_tile * quantile_embedding  # [ (batch x tau) x feature ]

        if not self.dueling:
            q_net = rearrange(
                hk.Sequential(
                    [
                        self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                        for i in range(2 * self.hidden_n)
                    ]
                    + [self.layer(self.action_size[0])]
                )(mul_embedding),
                "(b t) a -> b a t",
                b=batch_size,
                t=quaitle_shape[1],
            )  # [ batch x action x tau ]
            return q_net
        else:
            v = rearrange(
                hk.Sequential(
                    [
                        self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                        for i in range(2 * self.hidden_n)
                    ]
                    + [self.layer(1)]
                )(mul_embedding),
                "(b t) o -> b o t",
                b=batch_size,
                t=quaitle_shape[1],
            )  # [ batch x 1 x tau ]
            a = rearrange(
                hk.Sequential(
                    [
                        self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                        for i in range(2 * self.hidden_n)
                    ]
                    + [self.layer(self.action_size[0])]
                )(mul_embedding),
                "(b t) a -> b a t",
                b=batch_size,
                t=quaitle_shape[1],
            )  # [ batch x action x tau ]
            q = v + a - jnp.mean(a, axis=(1, 2), keepdims=True)
            return q


class FractionProposal(hk.Module):
    def __init__(self, support_size, node=256, hidden_n=2):
        super().__init__()
        self.support_size = support_size
        self.node = node
        self.hidden_n = hidden_n

    def __call__(self, feature):
        batch = feature.shape[0]
        log_probs = jax.nn.log_softmax(
            hk.Sequential(
                [
                    hk.Linear(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n - 1)
                ]
                + [hk.Linear(self.support_size)]
            )(feature),
            axis=1,
        )
        probs = jnp.exp(log_probs)
        tau_0 = jnp.zeros((batch, 1), dtype=np.float32)
        tau_1_N = jnp.cumsum(probs, axis=1)
        tau = jnp.concatenate((tau_0, tau_1_N), axis=1)
        tau_hat = jax.lax.stop_gradient((tau[:, :-1] + tau[:, 1:]) / 2.0)
        entropy = -jnp.sum(log_probs * probs, axis=-1, keepdims=True)
        return tau, tau_hat, entropy


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, n_support, policy_kwargs
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
        fqf = hk.transform(lambda x: FractionProposal(n_support)(x))
        model = hk.transform(
            lambda x, tau: Model(
                action_space, dueling=dueling_model, noisy=param_noise, **policy_kwargs
            )(x, tau)
        )

        preproc_fn = get_apply_fn_haiku_module(preproc)
        fqf_fn = get_apply_fn_haiku_module(fqf)
        model_fn = get_apply_fn_haiku_module(model)
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            pre_param = preproc.init(
                key1,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            out = preproc_fn(
                pre_param, key2, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            fqf_param = fqf.init(
                key3,
                out,
            )
            model_param = model.init(key4, out, fqf_fn(fqf_param, key4, out))
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
