import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.haiku.qnet.iqn_builder import Model
from model_builder.utils import dummy_observation, print_haiku_model_summary


class FractionProposal(hk.Module):
    def __init__(self, support_size, node=256, hidden_n=1):
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
                + [hk.Linear(self.support_size, w_init=hk.initializers.VarianceScaling(0.01))]
            )(feature),
            axis=1,
        )
        probs = jnp.exp(log_probs)
        tau_0 = jnp.zeros((batch, 1), dtype=np.float32)
        tau_1_N = jnp.cumsum(probs, axis=1)
        tau = jnp.concatenate((tau_0, tau_1_N), axis=1)  # [ batch x support_size + 1 ]
        tau_hat = jax.lax.stop_gradient(tau[:, :-1] + tau[:, 1:]) / 2.0  # [ batch x support_size ]
        entropy = -jnp.sum(log_probs * probs, axis=-1, keepdims=True)
        return tau, tau_hat, entropy


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, n_support, policy_kwargs
):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

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

        preproc_fn = preproc.apply
        fqf_fn = fqf.apply
        model_fn = model.apply
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            observation = dummy_observation(observation_space)
            pre_param = preproc.init(key1, observation)
            feature = preproc_fn(pre_param, key2, observation)
            fqf_param = fqf.init(key3, feature)
            _, tau_hat, _ = fqf_fn(fqf_param, key4, feature)
            model_param = model.init(key4, feature, tau_hat)
            params = hk.data_structures.merge(pre_param, model_param)
            print_haiku_model_summary(
                print_model,
                (preproc, observation),
                (fqf, feature),
                (model, feature, tau_hat),
            )
            return preproc_fn, model_fn, fqf_fn, params, fqf_param
        return preproc_fn, model_fn, fqf_fn

    return _model_builder
