import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.Module import PreProcess
from model_builder.model_config import ACTIVATIONS, DEFAULT_MLP, LayerConfig, MLPConfig
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_haiku_model_summary,
    split_actor_critic_kwargs,
)


def avgl1norm(x, epsilon=1e-6):
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True) + epsilon)


class Encoder(hk.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.width = width

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        return avgl1norm(
            hk.Sequential(
                [
                    hk.Linear(self.width),
                    jax.nn.elu,
                    hk.Linear(self.width),
                    jax.nn.elu,
                    hk.Linear(self.width),
                ]
            )(features)
        )


class Action_Encoder(hk.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.width = width

    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return hk.Sequential(
            [
                hk.Linear(self.width),
                jax.nn.elu,
                hk.Linear(self.width),
                jax.nn.elu,
                hk.Linear(self.width),
            ]
        )(jnp.concatenate([zs, action], axis=1))


class Actor(hk.Module):
    def __init__(self, action_size, network: MLPConfig = DEFAULT_MLP):
        super().__init__()
        self.action_size = action_size
        self.network = network
        self.layer = hk.Linear

    def __call__(self, features: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        a0 = avgl1norm(self.layer(self.network.layers[0].units)(features))
        embed_concat = jnp.concatenate([a0, zs], axis=1)
        return hk.Sequential(
            [
                stage
                for layer in self.network.layers
                for stage in (self.layer(layer.units), ACTIVATIONS[layer.activation])
            ]
            + [
                self.layer(
                    self.action_size[0],
                    w_init=hk.initializers.RandomUniform(-0.03, 0.03),
                ),
                jax.nn.tanh,
            ]
        )(embed_concat)


class Critic(hk.Module):
    def __init__(self, network: MLPConfig | None = None):
        super().__init__()
        self.network = MLPConfig((LayerConfig(256, "elu"),) * 2) if network is None else network
        self.layer = hk.Linear

    def __call__(
        self,
        features: jnp.ndarray,
        zs: jnp.ndarray,
        zsa: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        embedding = jnp.concatenate([zs, zsa], axis=1)
        q0 = avgl1norm(self.layer(self.network.layers[0].units)(concat))
        embed_concat = jnp.concatenate([q0, embedding], axis=1)
        return hk.Sequential(
            [
                stage
                for layer in self.network.layers
                for stage in (self.layer(layer.units), ACTIVATIONS[layer.activation])
            ]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(embed_concat)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        policy_kwargs,
        critic_default=MLPConfig((LayerConfig(256, "elu"),) * 2),
        allowed_embeddings=("normal",),
    )
    if not actor_kwargs["network"].layers or not critic_kwargs["network"].layers:
        raise ValueError("TD7 actor_model and critic_model require at least one hidden layer")

    def model_builder(key=None, print_model=False):
        def encode_state(obs, role, network: MLPConfig, shared_features=None):
            feature = PreProcess(
                observation_space, embedding_mode=network.embedding_mode, role=role
            )(obs, shared_features)
            return feature, Encoder(width=network.layers[0].units)(feature)

        actor_encoder = hk.transform(
            lambda obs: encode_state(obs, "actor", actor_kwargs["network"])
        )
        critic_encoder = hk.transform(
            lambda obs, shared_features: encode_state(
                obs, "critic", critic_kwargs["network"], shared_features
            )
        )
        shared_preproc = hk.transform(
            lambda obs: PreProcess(
                observation_space,
                embedding_mode=actor_kwargs["network"].embedding_mode,
                role="actor",
            ).shared_features(obs)
        )
        actor_action_encoder = hk.transform(
            lambda zs, actions: Action_Encoder(width=actor_kwargs["network"].layers[0].units)(
                zs, actions
            )
        )
        critic_action_encoder = hk.transform(
            lambda zs, actions: Action_Encoder(width=critic_kwargs["network"].layers[0].units)(
                zs, actions
            )
        )
        actor = hk.transform(lambda feature, zs: Actor(action_size, **actor_kwargs)(feature, zs))
        critic = hk.transform(
            lambda feature, zs, zsa, actions: (
                Critic(**critic_kwargs)(feature, zs, zsa, actions),
                Critic(**critic_kwargs)(feature, zs, zsa, actions),
            )
        )
        functions = (
            actor_encoder.apply,
            get_critic_apply_fn(critic_encoder.apply, shared_preproc.apply),
            actor_action_encoder.apply,
            critic_action_encoder.apply,
            actor.apply,
            critic.apply,
        )
        if key is None:
            return functions
        keys = jax.random.split(key, 6)
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_encoder_params = actor_encoder.init(keys[0], observation)
        shared_features = shared_preproc.apply(actor_encoder_params, None, observation)
        critic_encoder_params = critic_encoder.init(keys[1], observation, shared_features)
        actor_feature, actor_zs = actor_encoder.apply(actor_encoder_params, None, observation)
        critic_feature, critic_zs = functions[1](
            critic_encoder_params, actor_encoder_params, None, observation
        )
        actor_encoder_params = hk.data_structures.merge(
            actor_encoder_params, actor_action_encoder.init(keys[2], actor_zs, action)
        )
        critic_encoder_params = hk.data_structures.merge(
            critic_encoder_params,
            critic_action_encoder.init(keys[3], critic_zs, action),
        )
        critic_zsa = critic_action_encoder.apply(critic_encoder_params, None, critic_zs, action)
        policy_params = actor.init(keys[4], actor_feature, actor_zs)
        critic_params = critic.init(keys[5], critic_feature, critic_zs, critic_zsa, action)
        print_haiku_model_summary(
            print_model,
            (actor_encoder, observation),
            (critic_encoder, observation, shared_features),
            (actor_action_encoder, actor_zs, action),
            (critic_action_encoder, critic_zs, action),
            (actor, actor_feature, actor_zs),
            (critic, critic_feature, critic_zs, critic_zsa, action),
        )
        return (
            *functions,
            actor_encoder_params,
            critic_encoder_params,
            policy_params,
            critic_params,
        )

    return model_builder
