import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.dpg.ddpg_td3_blocks import GaussianActor
from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_haiku_model_summary,
    split_actor_critic_kwargs,
)


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2, support_n=200):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.support_n = support_n
        self.layer = hk.Linear

    def __call__(self, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(self.support_n, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(concat)


def model_builder_maker(observation_space, action_size, support_n, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        def actor_forward(observation):
            feature = PreProcess(observation_space, embedding_mode=embedding_mode, role="actor")(
                observation
            )
            return GaussianActor(action_size, **actor_kwargs)(feature)

        def shared_forward(observation):
            return PreProcess(
                observation_space, embedding_mode=embedding_mode, role="actor"
            ).shared_features(observation)

        def critic_forward(observation, shared_features, action):
            feature = PreProcess(observation_space, embedding_mode=embedding_mode, role="critic")(
                observation, shared_features
            )
            return (
                Critic(support_n=support_n, **critic_kwargs)(feature, action),
                Critic(support_n=support_n, **critic_kwargs)(feature, action),
            )

        actor = hk.transform(actor_forward)
        critic = hk.transform(critic_forward)
        shared_preproc = hk.transform(shared_forward)
        critic_fn = get_critic_apply_fn(critic.apply, shared_preproc.apply)
        if key is None:
            return actor.apply, critic_fn
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor.init(actor_key, observation)
        shared_features = shared_preproc.apply(policy_params, None, observation)
        critic_params = critic.init(critic_key, observation, shared_features, action)
        print_haiku_model_summary(
            print_model,
            (actor, observation),
            (critic, observation, shared_features, action),
        )
        return actor.apply, critic_fn, policy_params, critic_params

    return model_builder
