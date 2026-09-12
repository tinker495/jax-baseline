import haiku as hk
import jax
import numpy as np

from model_builder.haiku.dpg.ddpg_td3_blocks import Critic, GaussianActor
from model_builder.haiku.Module import PreProcess
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_haiku_model_summary,
    split_actor_critic_kwargs,
)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        policy_kwargs, allowed_embeddings=("normal",)
    )

    def model_builder(key=None, print_model=False):
        def actor_forward(observation):
            feature = PreProcess(
                observation_space,
                embedding_mode=actor_kwargs["network"].embedding_mode,
                role="actor",
            )(observation)
            return GaussianActor(action_size, **actor_kwargs)(feature)

        def shared_forward(observation):
            return PreProcess(
                observation_space,
                embedding_mode=actor_kwargs["network"].embedding_mode,
                role="actor",
            ).shared_features(observation)

        def critic_forward(observation, shared_features, action):
            feature = PreProcess(
                observation_space,
                embedding_mode=critic_kwargs["network"].embedding_mode,
                role="critic",
            )(observation, shared_features)
            return (
                Critic(**critic_kwargs)(feature, action),
                Critic(**critic_kwargs)(feature, action),
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
