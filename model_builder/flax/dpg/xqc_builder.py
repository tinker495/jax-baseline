import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_flax_model_summary


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 4

    def normalize(self, feature, training):
        return nn.BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(feature)

    @nn.compact
    def __call__(self, feature: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        feature = self.normalize(feature, training)
        for _ in range(self.hidden_n):
            feature = nn.Dense(
                self.node,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            )(feature)
            feature = self.normalize(feature, training)
            feature = jax.nn.relu(feature)
        mu = nn.Dense(
            self.action_size[0],
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
        )(feature)
        log_std = nn.Dense(
            self.action_size[0],
            kernel_init=nn.initializers.orthogonal(1.0),
        )(feature)
        return mu, jnp.clip(log_std, -10.0, 2.0)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 4
    n_atoms: int = 101

    def normalize(self, feature, training):
        return nn.BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(feature)

    @nn.compact
    def __call__(
        self, feature: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        feature = self.normalize(concat, training)
        for _ in range(self.hidden_n):
            feature = nn.Dense(
                self.node * 2,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            )(feature)
            feature = self.normalize(feature, training)
            feature = jax.nn.relu(feature)
        return nn.Dense(
            self.n_atoms,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
        )(feature)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {**policy_kwargs, "hidden_n": 4}
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

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
            observation = dummy_observation(observation_space)
            action = np.zeros((1, *action_size), dtype=np.float32)
            policy_params = model_actor.init(key, observation, True)
            feature = preproc_fn(policy_params, key, observation)
            critic_params = model_critic.init(key, feature, action, True)
            print_flax_model_summary(
                print_model,
                key,
                (model_actor, observation, True),
                (model_critic, feature, action, True),
            )
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
