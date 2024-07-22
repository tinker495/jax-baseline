import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_uniform_initializers
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        action = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(self.action_size[0], kernel_init=clip_uniform_initializers(-0.03, 0.03)),
                jax.nn.tanh,
            ]
        )(feature)
        return action


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))]
        )(concat)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Actor_Merged(nn.Module):
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

        actor_model = Actor_Merged()
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
                preproc_fn(policy_params, key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]),
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
