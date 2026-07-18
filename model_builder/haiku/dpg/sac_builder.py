import haiku as hk
import jax
import numpy as np

from model_builder.haiku.dpg.ddpg_td3_blocks import Critic, GaussianActor
from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_haiku_model_summary


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        actor = hk.transform(lambda x: GaussianActor(action_size, **policy_kwargs)(x))
        critic = hk.transform(
            lambda x, a: (
                Critic(**policy_kwargs)(x, a),
                Critic(**policy_kwargs)(x, a),
            )
        )
        preproc_fn = preproc.apply
        actor_fn = actor.apply
        critic_fn = critic.apply
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            observation = dummy_observation(observation_space)
            action = np.zeros((1, action_size[0]))
            pre_param = preproc.init(key1, observation)
            feature = preproc.apply(pre_param, key2, observation)
            actor_param = actor.init(key3, feature)
            critic_param = critic.init(key4, feature, action)

            params = hk.data_structures.merge(pre_param, actor_param, critic_param)
            print_haiku_model_summary(
                print_model,
                (preproc, observation),
                (actor, feature),
                (critic, feature, action),
            )
            return preproc_fn, actor_fn, critic_fn, params
        return preproc_fn, actor_fn, critic_fn

    return _model_builder
