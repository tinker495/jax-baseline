from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, SimbaV2Head, avgl1norm, network_body
from model_builder.flax.Module import PreProcess
from model_builder.model_config import (
    LayerConfig,
    MLPConfig,
    ModelConfig,
    ResidualConfig,
)
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Encoder(nn.Module):
    network: ModelConfig = MLPConfig()
    normalize: bool = True

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            return network_body(features, self.network)
        width = (
            self.network.layers[0].units
            if isinstance(self.network, MLPConfig)
            else self.network.blocks[0]
        )
        encoded = nn.Sequential([Dense(width), jax.nn.elu, Dense(width), jax.nn.elu, Dense(width)])(
            features
        )
        return avgl1norm(encoded) if self.normalize else encoded


class Actor(nn.Module):
    action_size: tuple
    network: ModelConfig = MLPConfig()
    layer: type[nn.Dense] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            base = network_body(features, self.network)
            encoded = network_body(jnp.concatenate([base, zs], axis=1), self.network)
            return jax.nn.tanh(SimbaV2Head(self.network.blocks[-1], self.action_size[0])(encoded))
        width = (
            self.network.layers[0].units
            if isinstance(self.network, MLPConfig)
            else self.network.blocks[0]
        )
        base = avgl1norm(self.layer(width)(features))
        encoded = network_body(jnp.concatenate([base, zs], axis=1), self.network, self.layer)
        return jax.nn.tanh(
            self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))(encoded)
        )


class Critic(nn.Module):
    network: ModelConfig = MLPConfig((LayerConfig(256, "elu"),) * 2)
    layer: type[nn.Dense] = Dense

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        zs: jnp.ndarray,
        zsa: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            base = network_body(concat, self.network)
            encoded = network_body(jnp.concatenate([base, zs, zsa], axis=1), self.network)
            return SimbaV2Head(self.network.blocks[-1], 1)(encoded)
        width = (
            self.network.layers[0].units
            if isinstance(self.network, MLPConfig)
            else self.network.blocks[0]
        )
        base = avgl1norm(self.layer(width)(concat))
        encoded = network_body(jnp.concatenate([base, zs, zsa], axis=1), self.network, self.layer)
        return self.layer(1, kernel_init=clip_factorized_uniform(3))(encoded)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        policy_kwargs,
        critic_default=MLPConfig((LayerConfig(256, "elu"),) * 2),
        allowed_types=("mlp", "simba", "simbav2"),
    )
    for options in (actor_kwargs, critic_kwargs):
        if isinstance(options["network"], MLPConfig) and not options["network"].layers:
            raise ValueError("TD7 actor_model and critic_model require at least one hidden layer")

    def model_builder(key=None, print_model=False):
        class RoleEncoder(nn.Module):
            role: Literal["actor", "critic"]
            network: ModelConfig

            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=self.network.embedding_mode,
                    role=self.role,
                )
                self.enc = Encoder(network=self.network)
                self.act_enc = Encoder(network=self.network, normalize=False)

            def __call__(self, obs, actions, shared_features=None):
                feature, zs = self.encode_state(obs, shared_features)
                return feature, zs, self.encode_action(zs, actions)

            def encode_state(self, obs, shared_features=None):
                feature = self.preproc(obs, shared_features)
                return feature, self.enc(feature)

            def shared_features(self, obs):
                return self.preproc.shared_features(obs)

            def encode_action(self, zs, actions):
                return self.act_enc(jnp.concatenate([zs, actions], axis=1))

        class TwinCritic(nn.Module):
            def setup(self):
                self.crit1 = Critic(**critic_kwargs)
                self.crit2 = Critic(**critic_kwargs)

            def __call__(self, feature, zs, zsa, actions):
                return self.crit1(feature, zs, zsa, actions), self.crit2(feature, zs, zsa, actions)

        actor_encoder_model = RoleEncoder("actor", actor_kwargs["network"])
        critic_encoder_model = RoleEncoder("critic", critic_kwargs["network"])
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
