import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.dpg.gaussian_blocks import Actor
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    support_n: int = 25
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
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

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                mu, log_std = self.actor(feature)
                return mu, log_std

            def preprocess(self, x):
                x = self.preproc(x)
                return x

            def actor(self, x):
                return self.act(x)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = Critic(support_n=support_n, **policy_kwargs)
                self.crit2 = Critic(support_n=support_n, **policy_kwargs)

            def __call__(self, x, a):
                return (self.crit1(x, a), self.crit2(x, a))

        model_actor = Merged_Actor()
        preproc_fn = get_apply_fn_flax_module(model_actor, model_actor.preprocess)
        actor_fn = get_apply_fn_flax_module(model_actor, model_actor.actor)
        model_critic = Merged_Critic()
        critic_fn = get_apply_fn_flax_module(model_critic)
        if key is not None:
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            policy_params = model_actor.init(key, observation)
            feature = preproc_fn(policy_params, key, observation)
            critic_params = model_critic.init(key, feature, action)
            print_flax_model_summary(
                print_model,
                key,
                (model_actor, observation),
                (model_critic, feature, action),
            )
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
