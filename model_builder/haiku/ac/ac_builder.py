import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.haiku.Module import PreProcess, pop_embedding_mode
from model_builder.utils import dummy_observation, print_haiku_model_summary


class Actor(hk.Module):
    def __init__(self, action_size, action_type, node=256, hidden_n=2):
        super().__init__()
        self.action_size = action_size
        self.action_type = action_type
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        mlp = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        if self.action_type == "discrete":
            return self.layer(
                self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03)
            )(mlp)
        if self.action_type == "continuous":
            mu = self.layer(self.action_size[0], w_init=hk.initializers.RandomUniform(-0.03, 0.03))(
                mlp
            )
            log_std = hk.get_parameter(
                "log_std", [1, self.action_size[0]], jnp.float32, init=jnp.zeros
            )
            return mu, log_std


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        return hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(feature)


def model_builder_maker(observation_space, action_size, action_type, policy_kwargs):
    policy_kwargs, embedding_mode = pop_embedding_mode(policy_kwargs)

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: PreProcess(observation_space, embedding_mode=embedding_mode)(x)
        )
        actor = hk.transform(lambda x: Actor(action_size, action_type, **policy_kwargs)(x))
        critic = hk.transform(lambda x: Critic(**policy_kwargs)(x))
        preproc_fn = preproc.apply
        actor_fn = actor.apply
        critic_fn = critic.apply
        if key is not None:
            key1, key2, key3, key4 = jax.random.split(key, num=4)
            observation = dummy_observation(observation_space)
            pre_param = preproc.init(key1, observation)
            feature = preproc.apply(pre_param, key2, observation)
            actor_param = actor.init(key3, feature)
            critic_param = critic.init(key4, feature)

            params = hk.data_structures.merge(pre_param, actor_param, critic_param)
            print_haiku_model_summary(
                print_model,
                (preproc, observation),
                (actor, feature),
                (critic, feature),
            )
            return preproc_fn, actor_fn, critic_fn, params
        return preproc_fn, actor_fn, critic_fn

    return _model_builder
