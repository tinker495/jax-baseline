import flax.linen as nn
import jax
import jax.numpy as jnp

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Actor(nn.Module):
    action_size: list[int]
    action_type: str
    node: int
    hidden_n: int
    layer: type[nn.Module] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        mlp = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(features)
        if self.action_type == "discrete":
            action_probs = self.layer(
                self.action_size[0], kernel_init=clip_factorized_uniform(0.01)
            )(mlp)
            return action_probs
        elif self.action_type == "continuous":
            mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(0.01))(mlp)
            log_std = jnp.clip(
                self.param("log_std", nn.initializers.zeros, (1, self.action_size[0])),
                -20,
                2,
            )
            return mu, log_std
        raise ValueError(f"Unsupported action type: {self.action_type}")


class Critic(nn.Module):
    node: int
    hidden_n: int
    layer: type[nn.Module] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_factorized_uniform(0.01))]
        )(features)
        return net


def model_builder_maker(observation_space, action_size, action_type, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        class ActorModel(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=embedding_mode,
                    role="actor",
                    name="PreProcess_0",
                )
                self.act = Actor(action_size, action_type, **actor_kwargs, name="Actor_0")

            def __call__(self, x):
                return self.act(self.preproc(x))

            def shared_features(self, x):
                return self.preproc.shared_features(x)

        class CriticModel(nn.Module):
            @nn.compact
            def __call__(self, x, shared_features):
                return Critic(**critic_kwargs)(
                    PreProcess(observation_space, embedding_mode=embedding_mode, role="critic")(
                        x, shared_features
                    )
                )

        actor = ActorModel()
        critic = CriticModel()
        actor_fn = get_apply_fn_flax_module(actor)
        shared_fn = get_apply_fn_flax_module(actor, method=actor.shared_features)
        critic_fn = get_critic_apply_fn(get_apply_fn_flax_module(critic), shared_fn)
        if key is not None:
            actor_key, critic_key = jax.random.split(key)
            observation = dummy_observation(observation_space)
            actor_params = actor.init(actor_key, observation)
            shared_features = shared_fn(actor_params, None, observation)
            critic_params = critic.init(critic_key, observation, shared_features)
            print_flax_model_summary(
                print_model, key, (actor, observation), (critic, observation, shared_features)
            )
            return actor_fn, critic_fn, actor_params, critic_params
        return actor_fn, critic_fn

    return _model_builder
