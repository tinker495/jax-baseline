import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.layers import SimbaV2Block, SimbaV2Embedding, SimbaV2Head
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Encoder(nn.Module):
    node: int = 256
    hidden_n: int = 3

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoded = SimbaV2Embedding(self.node)(feature)
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
    def __call__(self, feature: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        base = SimbaV2Embedding(self.node)(feature)
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
        self, feature: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
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

    def model_builder(key=None, print_model=False):
        class Merge_encoder(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.enc = Encoder(**policy_kwargs)
                self.act_enc = ActionEncoder(**policy_kwargs)

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
                return self.crit1(feature, zs, zsa, actions), self.crit2(feature, zs, zsa, actions)

        encoder_model = Merge_encoder()
        policy_model = Actor(action_size=action_size, **policy_kwargs)
        critic_model = Merged_Critic()

        preproc_fn = get_apply_fn_flax_module(encoder_model, encoder_model.preprocess)
        encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.encoder)
        action_encoder_fn = get_apply_fn_flax_module(encoder_model, encoder_model.action_encoder)
        actor_fn = get_apply_fn_flax_module(policy_model)
        critic_fn = get_apply_fn_flax_module(critic_model)

        if key is not None:
            zero_obs = dummy_observation(observation_space)
            zero_action = np.zeros((1, *action_size), dtype=np.float32)
            encoder_params = encoder_model.init(key, zero_obs, zero_action)
            feature, zs = encoder_model.apply(
                encoder_params,
                zero_obs,
                method=encoder_model.feature_and_zs,
            )
            policy_params = policy_model.init(key, feature, zs)
            feature_full, zs_full, zsa_full = encoder_model.apply(
                encoder_params,
                zero_obs,
                zero_action,
            )
            critic_params = critic_model.init(
                key,
                feature_full,
                zs_full,
                zsa_full,
                zero_action,
            )
            print_flax_model_summary(
                print_model,
                key,
                (encoder_model, zero_obs, zero_action),
                (policy_model, feature, zs),
                (critic_model, feature_full, zs_full, zsa_full, zero_action),
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
