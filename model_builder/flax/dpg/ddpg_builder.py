import flax.linen as nn
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.dpg.ddpg_td3_blocks import Actor, Critic
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


def _make_model_builder(
    observation_space,
    action_size,
    policy_kwargs,
    *,
    actor_cls,
    critic_cls,
    twin_critic,
):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = actor_cls(action_size, **policy_kwargs)

            def __call__(self, x):
                return self.actor(self.preprocess(x))

            def preprocess(self, x):
                return self.preproc(x)

            def actor(self, x):
                return self.act(x)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = critic_cls(**policy_kwargs)
                self.crit2 = critic_cls(**policy_kwargs)

            def __call__(self, x, a):
                return self.crit1(x, a), self.crit2(x, a)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic() if twin_critic else critic_cls(**policy_kwargs)
        preproc_fn = get_apply_fn_flax_module(actor_model, actor_model.preprocess)
        actor_fn = get_apply_fn_flax_module(actor_model, actor_model.actor)
        critic_fn = get_apply_fn_flax_module(critic_model)
        if key is not None:
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            policy_params = actor_model.init(key, observation)
            feature = preproc_fn(policy_params, key, observation)
            critic_params = critic_model.init(key, feature, action)
            print_flax_model_summary(
                print_model,
                key,
                (actor_model, observation),
                (critic_model, feature, action),
            )
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        return preproc_fn, actor_fn, critic_fn

    return model_builder


def model_builder_maker(observation_space, action_size, policy_kwargs):
    return _make_model_builder(
        observation_space,
        action_size,
        policy_kwargs,
        actor_cls=Actor,
        critic_cls=Critic,
        twin_critic=False,
    )
