import haiku as hk
import jax
import numpy as np

from model_builder.haiku.dpg.ddpg_td3_blocks import Actor, Critic
from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    print_haiku_model_summary,
    split_actor_critic_kwargs,
)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs)

    def model_builder(key=None, print_model=False):
        def actor_forward(observation):
            feature = PreProcess(observation_space, embedding_mode=embedding_mode, role="actor")(
                observation
            )
            return Actor(action_size, **actor_kwargs)(feature)

        def critic_forward(observation, action):
            feature = PreProcess(observation_space, embedding_mode=embedding_mode, role="critic")(
                observation
            )
            return (
                Critic(**critic_kwargs)(feature, action),
                Critic(**critic_kwargs)(feature, action),
            )

        actor = hk.transform(actor_forward)
        critic = hk.transform(critic_forward)
        if key is None:
            return actor.apply, critic.apply
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor.init(actor_key, observation)
        critic_params = critic.init(critic_key, observation, action)
        print_haiku_model_summary(
            print_model,
            (actor, observation),
            (critic, observation, action),
        )
        return actor.apply, critic.apply, policy_params, critic_params

    return model_builder
