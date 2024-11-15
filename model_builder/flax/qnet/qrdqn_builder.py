import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, NoisyDense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


class Model(nn.Module):
    action_size: int
    node: int
    hidden_n: int
    noisy: bool
    dueling: bool
    support_n: int

    def setup(self) -> None:
        if not self.noisy:
            self.layer = Dense
        else:
            self.layer = NoisyDense

        if self.dueling:
            self.value_support_n = self.support_n // 2
            self.advantage_support_n = self.support_n - self.value_support_n

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.support_n,
                        kernel_init=clip_factorized_uniform(3 / self.support_n),
                    ),
                    lambda x: jnp.reshape(x, (x.shape[0], self.action_size[0], self.support_n)),
                ]
            )(feature)
            return q_net
        else:
            v = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.value_support_n,
                        kernel_init=clip_factorized_uniform(3 / self.support_n),
                    ),
                    lambda x: jnp.reshape(x, (x.shape[0], 1, self.value_support_n)),
                ]
            )(feature)
            v = jnp.tile(v, (1, self.action_size[0], 1))
            a = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.advantage_support_n,
                        kernel_init=clip_factorized_uniform(3 / self.support_n),
                    ),
                    lambda x: jnp.reshape(
                        x, (x.shape[0], self.action_size[0], self.advantage_support_n)
                    ),
                ]
            )(feature)
            # q = batch x action x support / support = [v1, a1, v2, a2, ...]
            # Reshape v and a to interleave them along support axis
            v_reshaped = jnp.reshape(v, (v.shape[0], v.shape[1], -1, 1))  # [B, A, S/2, 1]
            a_reshaped = jnp.reshape(a, (a.shape[0], a.shape[1], -1, 1))  # [B, A, S/2, 1]
            q = jnp.reshape(
                jnp.concatenate([v_reshaped, a_reshaped], axis=3),  # [B, A, S/2, 2]
                (v.shape[0], v.shape[1], -1),  # [B, A, S]
            )
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

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.qnet = Model(
                    action_space,
                    dueling=dueling_model,
                    noisy=param_noise,
                    support_n=support_n,
                    **policy_kwargs
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
            params = model.init(
                key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", params)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, params
        else:
            return preproc_fn, model_fn

    return model_builder
