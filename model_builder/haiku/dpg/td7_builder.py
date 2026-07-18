import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_haiku_model_summary


def avgl1norm(x, epsilon=1e-6):
    return x / (jnp.abs(x).mean(axis=-1, keepdims=True) + epsilon)


class Encoder(hk.Module):
    def __init__(self, node=256, hidden_n=3):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoder = hk.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(feature)
        return avgl1norm(encoder)


class Action_Encoder(hk.Module):
    def __init__(self, node=256, hidden_n=3):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([zs, action], axis=1)
        return hk.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(concat)


class Actor(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        a0 = avgl1norm(self.layer(self.node)(feature))
        embed_concat = jnp.concatenate([a0, zs], axis=1)
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)),
                jax.nn.tanh,
            ]
        )(embed_concat)


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(
        self, feature: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        embedding = jnp.concatenate([zs, zsa], axis=1)
        q0 = avgl1norm(self.layer(self.node)(concat))
        embed_concat = jnp.concatenate([q0, embedding], axis=1)
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.elu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(embed_concat)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        encoder = hk.transform(lambda x: Encoder()(x))
        action_encoder = hk.transform(lambda zs, a: Action_Encoder()(zs, a))
        actor = hk.transform(lambda x, zs: Actor(action_size, **policy_kwargs)(x, zs))
        critic = hk.transform(
            lambda x, zs, zsa, a: (
                Critic(**policy_kwargs)(x, zs, zsa, a),
                Critic(**policy_kwargs)(x, zs, zsa, a),
            )
        )
        preproc_fn = preproc.apply
        encoder_fn = encoder.apply
        action_encoder_fn = action_encoder.apply
        actor_fn = actor.apply
        critic_fn = critic.apply
        if key is not None:
            keys = jax.random.split(key, num=9)
            observation = dummy_observation(observation_space)
            action = np.zeros((1, action_size[0]))
            pre_param = preproc.init(keys[0], observation)
            feature = preproc.apply(pre_param, keys[1], observation)

            encoder_param = encoder.init(keys[2], feature)
            zs = encoder.apply(encoder_param, keys[3], feature)

            action_encoder_param = action_encoder.init(keys[4], zs, action)
            zsa = action_encoder.apply(action_encoder_param, keys[5], zs, action)

            actor_param = actor.init(keys[6], feature, zs)
            critic_param = critic.init(keys[7], feature, zs, zsa, action)

            encoder_params = hk.data_structures.merge(
                pre_param, encoder_param, action_encoder_param
            )
            params = hk.data_structures.merge(actor_param, critic_param)
            print_haiku_model_summary(
                print_model,
                (preproc, observation),
                (encoder, feature),
                (action_encoder, zs, action),
                (actor, feature, zs),
                (critic, feature, zs, zsa, action),
            )
            return (
                preproc_fn,
                encoder_fn,
                action_encoder_fn,
                actor_fn,
                critic_fn,
                encoder_params,
                params,
            )
        return preproc_fn, encoder_fn, action_encoder_fn, actor_fn, critic_fn

    return _model_builder
