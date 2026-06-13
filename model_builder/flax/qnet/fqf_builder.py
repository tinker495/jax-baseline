import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.flax.qnet.iqn_builder import Model
from model_builder.utils import print_param


class FractionProposal(nn.Module):
    support_size: int
    node: int = 256
    hidden_n: int = 1

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        batch = feature.shape[0]
        log_probs = jax.nn.log_softmax(
            nn.Sequential(
                [Dense(self.node) if i % 2 == 0 else nn.relu for i in range(2 * self.hidden_n)]
                + [
                    Dense(
                        self.support_size,
                        kernel_init=jax.nn.initializers.variance_scaling(
                            0.01, mode="fan_in", distribution="normal"
                        ),
                    )
                ],
            )(feature),
            axis=-1,
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
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, pre_postprocess=nn.Dense(512)
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
        fqf = FractionProposal(n_support)
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        fqf_fn = get_apply_fn_flax_module(fqf)
        model_fn = get_apply_fn_flax_module(model, model.q)
        if key is not None:
            tau = jax.random.uniform(key, (1, 2))
            params = model.init(
                key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space], tau
            )
            out = preproc_fn(
                params, key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            fqf_param = fqf.init(key, out)
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("iqn", params)
                print_param("fqf", fqf_param)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, fqf_fn, params, fqf_param
        else:
            return preproc_fn, model_fn, fqf_fn

    return model_builder
