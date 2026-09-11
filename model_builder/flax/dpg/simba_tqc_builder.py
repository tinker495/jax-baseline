import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import LOG_STD_MEAN, LOG_STD_SCALE, Dense, ResidualBlock
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        linear = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [
                nn.LayerNorm(),
            ]
        )(features)
        mu = Dense(
            self.action_size[0],
            kernel_init=clip_factorized_uniform(3),
        )(linear)
        log_std = Dense(
            self.action_size[0],
            kernel_init=clip_factorized_uniform(3),
            bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
        )(linear)  # initialize std with high values
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std / LOG_STD_SCALE)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    support_n: int = 200

    @nn.compact
    def __call__(self, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        q_net = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [
                nn.LayerNorm(),
                Dense(
                    self.support_n,
                    kernel_init=clip_factorized_uniform(3 / self.support_n),
                ),
            ]
        )(concat)
        return q_net


def model_builder_maker(observation_space, action_size, support_n, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role="actor"
                )
                self.act = Actor(action_size, **actor_kwargs)

            def __call__(self, observation):
                return self.act(self.preproc(observation))

            def shared_features(self, observation):
                return self.preproc.shared_features(observation)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role="critic"
                )
                self.crit1 = Critic(support_n=support_n, **critic_kwargs)
                self.crit2 = Critic(support_n=support_n, **critic_kwargs)

            def __call__(self, observation, shared_features, action):
                feature = self.preproc(observation, shared_features)
                return self.crit1(feature, action), self.crit2(feature, action)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model)
        shared_preproc_fn = get_apply_fn_flax_module(
            actor_model, method=actor_model.shared_features
        )
        critic_fn = get_critic_apply_fn(get_apply_fn_flax_module(critic_model), shared_preproc_fn)
        if key is None:
            return actor_fn, critic_fn
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor_model.init(actor_key, observation)
        shared_features = shared_preproc_fn(policy_params, None, observation)
        critic_params = critic_model.init(critic_key, observation, shared_features, action)
        print_flax_model_summary(
            print_model,
            key,
            (actor_model, observation),
            (critic_model, observation, shared_features, action),
        )
        return actor_fn, critic_fn, policy_params, critic_params

    return model_builder
