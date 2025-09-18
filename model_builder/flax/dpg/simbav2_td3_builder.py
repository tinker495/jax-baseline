import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.layers import SimbaV2Block, SimbaV2Embedding, SimbaV2Head
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        encoded = SimbaV2Embedding(self.node)(feature)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        logits = SimbaV2Head(self.node, self.action_size[0])(encoded)
        return jax.nn.tanh(logits)


class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        encoded = SimbaV2Embedding(self.node)(concat)
        for _ in range(self.hidden_n):
            encoded = SimbaV2Block(self.node)(encoded)
        q_value = SimbaV2Head(self.node, 1)(encoded)
        return q_value


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class ActorMerged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                return self.actor(feature)

            def preprocess(self, x):
                return self.preproc(x)

            def actor(self, x):
                return self.act(x)

        class CriticsMerged(nn.Module):
            def setup(self):
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, x, a):
                return self.crit1(x, a), self.crit2(x, a)

        actor_model = ActorMerged()
        critics_model = CriticsMerged()
        preproc_fn = get_apply_fn_flax_module(actor_model, actor_model.preprocess)
        actor_fn = get_apply_fn_flax_module(actor_model, actor_model.actor)
        critic_fn = get_apply_fn_flax_module(critics_model)

        if key is not None:
            policy_params = actor_model.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            critic_params = critics_model.init(
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
