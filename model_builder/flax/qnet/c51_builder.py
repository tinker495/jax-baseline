import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, NoisyDense
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Model(nn.Module):
    action_size: int
    node: int
    hidden_n: int
    noisy: bool
    dueling: bool
    categorial_bar_n: int

    def setup(self) -> None:
        if not self.noisy:
            self.layer = Dense
        else:
            self.layer = NoisyDense

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
                        self.action_size[0] * self.categorial_bar_n,
                        kernel_init=clip_factorized_uniform(0.01),
                    ),
                    lambda x: jnp.reshape(
                        x, (x.shape[0], self.action_size[0], self.categorial_bar_n)
                    ),
                ]
            )(feature)
            return jax.nn.softmax(q_net, axis=2)
        else:
            v = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(self.categorial_bar_n, kernel_init=clip_factorized_uniform(0.01)),
                    lambda x: jnp.reshape(x, (x.shape[0], 1, self.categorial_bar_n)),
                ]
            )(feature)
            a = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.categorial_bar_n,
                        kernel_init=clip_factorized_uniform(0.01),
                    ),
                    lambda x: jnp.reshape(
                        x, (x.shape[0], self.action_size[0], self.categorial_bar_n)
                    ),
                ]
            )(feature)
            q = v + a - jnp.mean(a, axis=1, keepdims=True)
            return jax.nn.softmax(q, axis=2)


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.qnet = Model(
                    action_space,
                    dueling=dueling_model,
                    noisy=param_noise,
                    categorial_bar_n=categorial_bar_n,
                    **policy_kwargs,
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
            observation = dummy_observation(observation_space)
            params = model.init(key, observation)
            print_flax_model_summary(print_model, key, (model, observation))
            return preproc_fn, model_fn, params
        else:
            return preproc_fn, model_fn

    return model_builder
