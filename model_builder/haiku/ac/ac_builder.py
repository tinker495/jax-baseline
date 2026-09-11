import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_haiku_model_summary,
    split_actor_critic_kwargs,
)


class Actor(hk.Module):
    def __init__(self, action_size, action_type, node=256, hidden_n=2):
        super().__init__()
        self.action_size = action_size
        self.action_type = action_type
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        mlp = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(features)
        if self.action_type == "discrete":
            return self.layer(
                self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
            )(mlp)
        if self.action_type == "continuous":
            mu = self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03))(
                mlp
            )
            log_std = hk.get_parameter(
                "log_std", [1, self.action_size[0]], jnp.float32, init=jnp.zeros
            )
            return mu, log_std
        raise ValueError(f"Unsupported action type: {self.action_type}")


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(features)


def model_builder_maker(observation_space, action_size, action_type, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        actor = hk.transform(
            lambda x: Actor(action_size, action_type, **actor_kwargs)(
                PreProcess(observation_space, embedding_mode=embedding_mode, role="actor")(x)
            )
        )
        shared = hk.transform(
            lambda x: PreProcess(
                observation_space, embedding_mode=embedding_mode, role="actor"
            ).shared_features(x)
        )
        critic = hk.transform(
            lambda x, shared_features: Critic(**critic_kwargs)(
                PreProcess(observation_space, embedding_mode=embedding_mode, role="critic")(
                    x, shared_features
                )
            )
        )
        actor_fn = actor.apply
        critic_fn = get_critic_apply_fn(critic.apply, shared.apply)
        if key is not None:
            actor_key, critic_key = jax.random.split(key)
            observation = dummy_observation(observation_space)
            actor_params = actor.init(actor_key, observation)
            shared_features = shared.apply(actor_params, None, observation)
            critic_params = critic.init(critic_key, observation, shared_features)
            print_haiku_model_summary(
                print_model,
                (actor, observation),
                (critic, observation, shared_features),
            )
            return actor_fn, critic_fn, actor_params, critic_params
        return actor_fn, critic_fn

    return _model_builder
