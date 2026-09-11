import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.dpg.gaussian_blocks import Actor
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    support_n: int = 25
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(
                    self.support_n,
                    kernel_init=clip_factorized_uniform(3 / self.support_n),
                )
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

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role="critic"
                )
                self.crit1 = Critic(support_n=support_n, **critic_kwargs)
                self.crit2 = Critic(support_n=support_n, **critic_kwargs)

            def __call__(self, observation, action):
                feature = self.preproc(observation)
                return self.crit1(feature, action), self.crit2(feature, action)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model)
        critic_fn = get_apply_fn_flax_module(critic_model)
        if key is None:
            return actor_fn, critic_fn
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor_model.init(actor_key, observation)
        critic_params = critic_model.init(critic_key, observation, action)
        print_flax_model_summary(
            print_model,
            key,
            (actor_model, observation),
            (critic_model, observation, action),
        )
        return actor_fn, critic_fn, policy_params, critic_params

    return model_builder
