from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.layers import SimbaV2Block, SimbaV2Embedding, SimbaV2Head
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        encoded = SimbaV2Embedding(self.node)(features)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        return encoded


class ActionEncoder(nn.Module):
    node: int = 256
    hidden_n: int = 3

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([zs, action], axis=1)
        encoded = SimbaV2Embedding(self.node)(concat)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        return encoded


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, features: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        base = SimbaV2Embedding(self.node)(features)
        for _ in range(self.hidden_n):
            base = SimbaV2Block(self.node)(base)
        embed = jnp.concatenate([base, zs], axis=1)
        encoded = SimbaV2Embedding(self.node)(embed)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        action_logits = SimbaV2Head(self.node, self.action_size[0])(encoded)
        return jax.nn.tanh(action_logits)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        zs: jnp.ndarray,
        zsa: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        base = SimbaV2Embedding(self.node)(concat)
        for _ in range(self.hidden_n):
            base = SimbaV2Block(self.node)(base)
        embed = jnp.concatenate([base, zs, zsa], axis=1)
        encoded = SimbaV2Embedding(self.node)(embed)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        q_value = SimbaV2Head(self.node, 1)(encoded)
        return q_value


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class RoleEncoder(nn.Module):
            role: Literal["actor", "critic"]
            node: int
            hidden_n: int

            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role=self.role
                )
                self.enc = Encoder(node=self.node, hidden_n=self.hidden_n)
                self.act_enc = ActionEncoder(node=self.node, hidden_n=self.hidden_n)

            def __call__(self, obs, actions, shared_features=None):
                feature, zs = self.encode_state(obs, shared_features)
                return feature, zs, self.encode_action(zs, actions)

            def encode_state(self, obs, shared_features=None):
                feature = self.preproc(obs, shared_features)
                return feature, self.enc(feature)

            def shared_features(self, obs):
                return self.preproc.shared_features(obs)

            def encode_action(self, zs, actions):
                return self.act_enc(zs, actions)

        class TwinCritic(nn.Module):
            def setup(self):
                self.crit1 = Critic(**critic_kwargs)
                self.crit2 = Critic(**critic_kwargs)

            def __call__(self, feature, zs, zsa, actions):
                return self.crit1(feature, zs, zsa, actions), self.crit2(feature, zs, zsa, actions)

        actor_encoder_model = RoleEncoder(
            "actor",
            actor_kwargs["node"],
            {"hidden_n": 3, **actor_kwargs}["hidden_n"],
        )
        critic_encoder_model = RoleEncoder(
            "critic",
            critic_kwargs["node"],
            {"hidden_n": 3, **critic_kwargs}["hidden_n"],
        )
        policy_model = Actor(action_size=action_size, **actor_kwargs)
        critic_model = TwinCritic()
        shared_preproc_fn = get_apply_fn_flax_module(
            actor_encoder_model, method=actor_encoder_model.shared_features
        )
        functions = (
            get_apply_fn_flax_module(actor_encoder_model, actor_encoder_model.encode_state),
            get_critic_apply_fn(
                get_apply_fn_flax_module(critic_encoder_model, critic_encoder_model.encode_state),
                shared_preproc_fn,
            ),
            get_apply_fn_flax_module(actor_encoder_model, actor_encoder_model.encode_action),
            get_apply_fn_flax_module(critic_encoder_model, critic_encoder_model.encode_action),
            get_apply_fn_flax_module(policy_model),
            get_apply_fn_flax_module(critic_model),
        )
        if key is None:
            return functions
        actor_encoder_key, critic_encoder_key, actor_key, critic_key = jax.random.split(key, 4)
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_encoder_params = actor_encoder_model.init(actor_encoder_key, observation, action)
        shared_features = shared_preproc_fn(actor_encoder_params, None, observation)
        critic_encoder_params = critic_encoder_model.init(
            critic_encoder_key, observation, action, shared_features
        )
        actor_feature, actor_zs = functions[0](actor_encoder_params, None, observation)
        critic_feature, critic_zs = functions[1](
            critic_encoder_params, actor_encoder_params, None, observation
        )
        critic_zsa = functions[3](critic_encoder_params, None, critic_zs, action)
        policy_params = policy_model.init(actor_key, actor_feature, actor_zs)
        critic_params = critic_model.init(critic_key, critic_feature, critic_zs, critic_zsa, action)
        print_flax_model_summary(
            print_model,
            key,
            (actor_encoder_model, observation, action),
            (critic_encoder_model, observation, action, shared_features),
            (policy_model, actor_feature, actor_zs),
            (critic_model, critic_feature, critic_zs, critic_zsa, action),
        )
        return (
            *functions,
            actor_encoder_params,
            critic_encoder_params,
            policy_params,
            critic_params,
        )

    return model_builder
