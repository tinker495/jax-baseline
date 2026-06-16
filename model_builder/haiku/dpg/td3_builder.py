import haiku as hk
import jax
import numpy as np

from model_builder.haiku.apply import get_apply_fn_haiku_module
from model_builder.haiku.dpg.ddpg_td3_blocks import Actor, Critic
from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import print_param


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        actor = hk.transform(lambda x: Actor(action_size, **policy_kwargs)(x))
        critic = hk.transform(
            lambda x, a: (
                Critic(**policy_kwargs)(x, a),
                Critic(**policy_kwargs)(x, a),
            )
        )
        preproc_fn = get_apply_fn_haiku_module(preproc)
        actor_fn = get_apply_fn_haiku_module(actor)
        critic_fn = get_apply_fn_haiku_module(critic)
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            pre_param = preproc.init(
                key1,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            feature = preproc.apply(
                pre_param,
                key2,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            actor_param = actor.init(key3, feature)
            critic_param = critic.init(key4, feature, np.zeros((1, action_size[0])))

            params = hk.data_structures.merge(pre_param, actor_param, critic_param)
            if print_model:
                print("------------------build-haiku-model--------------------")
                print_param("preprocess", pre_param)
                print_param("actor", actor_param)
                print_param("critic", critic_param)
                print("-------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, params
        return preproc_fn, actor_fn, critic_fn

    return _model_builder
