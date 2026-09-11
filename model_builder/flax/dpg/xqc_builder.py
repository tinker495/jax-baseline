import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.Module import PreProcess, pop_embedding_mode
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 4

    def normalize(self, feature, training):
        return nn.BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(feature)

    @nn.compact
    def __call__(self, features: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        feature = features
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
    node: int = 512
    hidden_n: int = 4
    n_atoms: int = 101

    def normalize(self, feature, training):
        return nn.BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(feature)

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        feature = features
        concat = jnp.concatenate([feature, actions], axis=1)
        feature = self.normalize(concat, training)
        for _ in range(self.hidden_n):
            feature = nn.Dense(
                self.node,
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
    policy_kwargs = {**({} if policy_kwargs is None else policy_kwargs), "hidden_n": 4}
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(policy_kwargs, critic_node=512)

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role="actor"
                )
                self.act = Actor(action_size, **actor_kwargs)

            def __call__(self, observation, training: bool = True):
                return self.act(self.preproc(observation), training)

            def shared_features(self, observation):
                return self.preproc.shared_features(observation)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, role="critic"
                )
                self.crit1 = Critic(**critic_kwargs)
                self.crit2 = Critic(**critic_kwargs)

            def __call__(self, observation, shared_features, action, training: bool = True):
                feature = self.preproc(observation, shared_features)
                return self.crit1(feature, action, training), self.crit2(feature, action, training)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model, mutable=["batch_stats"])
        shared_preproc_fn = get_apply_fn_flax_module(
            actor_model, method=actor_model.shared_features
        )
        critic_fn = get_critic_apply_fn(
            get_apply_fn_flax_module(critic_model, mutable=["batch_stats"]),
            shared_preproc_fn,
        )
        if key is None:
            return actor_fn, critic_fn
        observation = dummy_observation(observation_space)
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor_model.init(actor_key, observation, True)
        shared_features = shared_preproc_fn(policy_params, None, observation)
        critic_params = critic_model.init(critic_key, observation, shared_features, action, True)
        print_flax_model_summary(
            print_model,
            key,
            (actor_model, observation, True),
            (critic_model, observation, shared_features, action, True),
        )
        return actor_fn, critic_fn, policy_params, critic_params

    return model_builder
