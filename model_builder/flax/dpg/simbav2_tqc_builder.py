import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.layers import (
    LOG_STD_MEAN,
    LOG_STD_SCALE,
    SimbaV2Block,
    SimbaV2Embedding,
    SimbaV2Head,
)
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoded = SimbaV2Embedding(self.node)(feature)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        mu = SimbaV2Head(
            self.node,
            self.action_size[0],
        )(encoded)
        log_std_raw = SimbaV2Head(
            self.node,
            self.action_size[0],
        )(encoded)
        log_std = LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std_raw / LOG_STD_SCALE)
        return mu, log_std


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    support_n: int = 200

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        encoded = SimbaV2Embedding(self.node)(concat)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        quantiles = SimbaV2Head(
            self.node,
            self.support_n,
        )(encoded)
        return quantiles


def model_builder_maker(observation_space, action_size, support_n, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                return self.actor(feature)

            def preprocess(self, x):
                return self.preproc(x)

            def actor(self, x):
                return self.act(x)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = Critic(support_n=support_n, **policy_kwargs)
                self.crit2 = Critic(support_n=support_n, **policy_kwargs)

            def __call__(self, x, a):
                return self.crit1(x, a), self.crit2(x, a)

        actor_model = Merged_Actor()
        critics_model = Merged_Critic()
        preproc_fn = get_apply_fn_flax_module(actor_model, actor_model.preprocess)
        actor_fn = get_apply_fn_flax_module(actor_model, actor_model.actor)
        critic_fn = get_apply_fn_flax_module(critics_model)

        if key is not None:
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            policy_params = actor_model.init(key, observation)
            feature = preproc_fn(policy_params, key, observation)
            critic_params = critics_model.init(key, feature, action)
            print_flax_model_summary(
                print_model,
                key,
                (actor_model, observation),
                (critics_model, feature, action),
            )
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
