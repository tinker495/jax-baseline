import flax.linen as nn
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.dpg.ddpg_td3_blocks import Actor, Critic
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import print_param


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                action = self.actor(feature)
                return action

            def preprocess(self, x):
                x = self.preproc(x)
                return x

            def actor(self, x):
                return self.act(x)

        actor_model = Merged_Actor()
        critic_model = Critic(**policy_kwargs)
        preproc_fn = get_apply_fn_flax_module(actor_model, actor_model.preprocess)
        actor_fn = get_apply_fn_flax_module(actor_model, actor_model.actor)
        critic_fn = get_apply_fn_flax_module(critic_model)
        if key is not None:
            policy_params = actor_model.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            critic_params = critic_model.init(
                key,
                preproc_fn(
                    policy_params,
                    key,
                    [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                ),
                np.zeros((1, *action_size), dtype=np.float32),
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", policy_params)
                print_param("", critic_params)
                print("------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
