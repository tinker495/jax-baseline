import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.apply import get_apply_fn_haiku_module
from model_builder.haiku.Module import PreProcess
from model_builder.utils import print_param

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN) / 2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN) / 2.0


class Actor(hk.Module):
    def __init__(self, action_size, node=256, hidden_n=2):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        linear = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [
                self.layer(
                    self.action_size[0] * 2, w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                )
            ]
        )(feature)
        mu, log_std = jnp.split(linear, 2, axis=-1)
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(
            log_std / LOG_STD_SCALE
        )  # jnp.clip(log_std,LOG_STD_MIN,LOG_STD_MAX)


class Critic(hk.Module):
    def __init__(self, node=256, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, w_init=hk.initializers.RandomUniform(-0.03, 0.03))]
        )(concat)
        return q_net


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

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
        else:
            return preproc_fn, actor_fn, critic_fn

    return _model_builder
