import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


class Actor(nn.Module):
    action_size: int
    action_type: str
    node: int
    hidden_n: int
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        mlp = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        if self.action_type == "discrete":
            action_probs = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(0.01))(
                mlp
            )
            return action_probs
        elif self.action_type == "continuous":
            mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(0.01))(mlp)
            log_std = self.param("log_std", jnp.zeros, (1, self.action_size[0]))
            return mu, log_std


class Critic(nn.Module):
    node: int
    hidden_n: int
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_factorized_uniform(0.01))]
        )(feature)
        return net


def model_builder_maker(observation_space, action_size, action_type, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def _model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, action_type, **policy_kwargs)
                self.cri = Critic(**policy_kwargs)

            def __call__(self, x):
                x = self.preproc(x)
                return self.act(x), self.cri(x)

            def preprocess(self, x):
                return self.preproc(x)

            def actor(self, x):
                return self.act(x)

            def critic(self, x):
                return self.cri(x)

        model = Merged()
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        actor_fn = get_apply_fn_flax_module(model, model.actor)
        critic_fn = get_apply_fn_flax_module(model, model.critic)
        if key is not None:
            params = model.init(
                key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", params)
                print("-------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, params
        else:
            return preproc_fn, actor_fn, critic_fn

    return _model_builder
