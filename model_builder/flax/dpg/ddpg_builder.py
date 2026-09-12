import flax.linen as nn
import jax
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.dpg.ddpg_td3_blocks import Actor, Critic
from model_builder.flax.Module import PreProcess
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


def _make_model_builder(observation_space, action_size, policy_kwargs, *, twin_critic):
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        policy_kwargs, allowed_types=("mlp", "simba", "simbav2")
    )

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=actor_kwargs["network"].embedding_mode,
                    role="actor",
                )
                self.act = Actor(action_size, **actor_kwargs)

            def __call__(self, x):
                return self.act(self.preproc(x))

            def shared_features(self, observation):
                return self.preproc.shared_features(observation)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=critic_kwargs["network"].embedding_mode,
                    role="critic",
                )
                self.crit1 = Critic(**critic_kwargs)
                if twin_critic:
                    self.crit2 = Critic(**critic_kwargs)

            def __call__(self, x, shared_features, a):
                feature = self.preproc(x, shared_features)
                if twin_critic:
                    return self.crit1(feature, a), self.crit2(feature, a)
                return self.crit1(feature, a)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model)
        shared_preproc_fn = get_apply_fn_flax_module(
            actor_model, method=actor_model.shared_features
        )
        critic_fn = get_critic_apply_fn(get_apply_fn_flax_module(critic_model), shared_preproc_fn)
        if key is not None:
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            actor_key, critic_key = jax.random.split(key)
            policy_params = actor_model.init(actor_key, observation)
            shared_features = shared_preproc_fn(policy_params, None, observation)
            critic_params = critic_model.init(critic_key, observation, shared_features, action)
            print_flax_model_summary(
                print_model,
                key,
                (actor_model, observation),
                (critic_model, observation, shared_features, action),
            )
            return actor_fn, critic_fn, policy_params, critic_params
        return actor_fn, critic_fn

    return model_builder


def model_builder_maker(observation_space, action_size, policy_kwargs):
    return _make_model_builder(
        observation_space,
        action_size,
        policy_kwargs,
        twin_critic=False,
    )
