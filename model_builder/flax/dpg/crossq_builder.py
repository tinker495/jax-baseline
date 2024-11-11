import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import Dense
from model_builder.flax.Module import BatchReNorm, PreProcess
from model_builder.utils import print_param

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN) / 2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN) / 2.0


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        linear = BatchReNorm(use_running_average=not training)(feature)
        for i in range(self.hidden_n):
            linear = self.layer(self.node)(linear)
            linear = BatchReNorm(use_running_average=not training)(linear)
            linear = jax.nn.relu(linear)
        mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(0.03))(linear)
        log_std = self.layer(
            self.action_size[0],
            kernel_init=clip_factorized_uniform(0.03),
            bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
        )(
            linear
        )  # initialize std with high values
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(
            log_std / LOG_STD_SCALE
        )  # jnp.clip(log_std,LOG_STD_MIN,LOG_STD_MAX)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(
        self, feature: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        feature = BatchReNorm(use_running_average=not training)(concat)
        for i in range(self.hidden_n):
            feature = self.layer(self.node)(feature)
            feature = BatchReNorm(use_running_average=not training)(feature)
            feature = jax.nn.tanh(feature)
        q_net = self.layer(1, kernel_init=clip_factorized_uniform(0.03))(feature)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x, training: bool = True):
                feature = self.preprocess(x)
                mu, log_std = self.actor(feature, training)
                return mu, log_std

            def preprocess(self, x):
                x = self.preproc(x)
                return x

            def actor(self, x, training: bool = True):
                return self.act(x, training)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, x, a, training: bool = True):
                return (self.crit1(x, a, training), self.crit2(x, a, training))

        model_actor = Merged_Actor()
        preproc_fn = get_apply_fn_flax_module(model_actor, model_actor.preprocess)
        actor_fn = get_apply_fn_flax_module(model_actor, model_actor.actor, mutable=["batch_stats"])
        model_critic = Merged_Critic()
        critic_fn = get_apply_fn_flax_module(model_critic, mutable=["batch_stats"])
        if key is not None:
            policy_params = model_actor.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                True,
            )
            critic_params = model_critic.init(
                key,
                preproc_fn(
                    policy_params,
                    key,
                    [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                ),
                np.zeros((1, *action_size), dtype=np.float32),
                True,
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
