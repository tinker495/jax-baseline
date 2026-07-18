import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense, ResidualBlock, avgl1norm
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoder = nn.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(feature)
        return avgl1norm(encoder)


class Action_Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([zs, action], axis=1)
        zsa = nn.Sequential(
            [
                self.layer(self.node) if i % 2 == 0 else jax.nn.elu
                for i in range(2 * self.hidden_n - 1)
            ]
        )(concat)
        return zsa


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        a0 = avgl1norm(Dense(self.node)(feature))
        embed_concat = jnp.concatenate([a0, zs], axis=1)
        action = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [
                nn.LayerNorm(),
                Dense(self.action_size[0], kernel_init=clip_factorized_uniform(3)),
                jax.nn.tanh,
            ]
        )(embed_concat)
        return action


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(
        self, feature: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        embedding = jnp.concatenate([zs, zsa], axis=1)
        q0 = avgl1norm(Dense(self.node)(concat))
        embed_concat = jnp.concatenate([q0, embedding], axis=1)
        q_net = nn.Sequential(
            [Dense(self.node)]
            + [ResidualBlock(self.node) for _ in range(self.hidden_n)]
            + [nn.LayerNorm(), Dense(1, kernel_init=clip_factorized_uniform(3))]
        )(embed_concat)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merge_encoder(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.enc = Encoder()
                self.act_enc = Action_Encoder()

            def __call__(self, x, a):
                feature = self.preprocess(x)
                zs = self.encoder(feature)
                zsa = self.action_encoder(zs, a)
                return feature, zs, zsa

            def preprocess(self, x):
                return self.preproc(x)

            def encoder(self, feature):
                return self.enc(feature)

            def action_encoder(self, zs, a):
                return self.act_enc(zs, a)

            def feature_and_zs(self, x):
                feature = self.preprocess(x)
                zs = self.encoder(feature)
                return feature, zs

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, feature, zs, zsa, actions):
                q = self.critic(feature, zs, zsa, actions)
                return q

            def critic(self, feature, zs, zsa, a):
                return (self.crit1(feature, zs, zsa, a), self.crit2(feature, zs, zsa, a))

        encoder_model = Merge_encoder()
        preproc_fn = get_apply_fn_flax_module(encoder_model, encoder_model.preprocess)
        encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.encoder)
        action_encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.action_encoder)
        policy_model = Actor(action_size=action_size)
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(policy_model)
        critic_fn = get_apply_fn_flax_module(critic_model)
        if key is not None:
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            encoder_params = encoder_model.init(key, observation, action)
            feature, zs, zsa = encoder_model.apply(encoder_params, observation, action)
            policy_params = policy_model.init(key, feature, zs)
            critic_params = critic_model.init(key, feature, zs, zsa, action)
            print_flax_model_summary(
                print_model,
                key,
                (encoder_model, observation, action),
                (policy_model, feature, zs),
                (critic_model, feature, zs, zsa, action),
            )
            return (
                preproc_fn,
                encoder_fn,
                action_encoder_fn,
                actor_fn,
                critic_fn,
                encoder_params,
                policy_params,
                critic_params,
            )
        else:
            return preproc_fn, encoder_fn, action_encoder_fn, actor_fn, critic_fn

    return model_builder
