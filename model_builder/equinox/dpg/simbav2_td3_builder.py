from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.dpg.simbav2_ddpg_builder import (
    SimbaV2Actor,
    SimbaV2Critic,
    _package_params,
)
from model_builder.equinox.Module import PreProcess
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param


class ActorWrapper(eqx.Module):
    preproc: PreProcess
    actor: SimbaV2Actor

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def actor_forward(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        return self.actor(feature, key=key)


class DoubleCritic(eqx.Module):
    critic1: SimbaV2Critic
    critic2: SimbaV2Critic

    def __call__(
        self,
        feature: jnp.ndarray,
        action: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        q1 = self.critic1(feature, action, key=key1)
        q2 = self.critic2(feature, action, key=key2)
        return q1, q2


def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_actor, key_c1, key_c2 = jax.random.split(rng, 4)

        preproc_module = PreProcess(observation_space, embedding_mode=embedding_mode, key=key_pre)
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_dim = preproc_module(dummy_obs).shape[-1]

        actor_core = SimbaV2Actor(feature_dim, action_size[0], node, hidden_n, key=key_actor)
        actor_wrapper = ActorWrapper(preproc_module, actor_core)
        actor_params_tree, actor_static = eqx.partition(actor_wrapper, eqx.is_array)
        actor_params = _package_params(actor_params_tree)
        preproc_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.preprocess)
        actor_fn = get_apply_fn_equinox_module(actor_static, actor_wrapper.actor_forward)

        critic_model = DoubleCritic(
            SimbaV2Critic(feature_dim, action_size[0], node, hidden_n, key=key_c1),
            SimbaV2Critic(feature_dim, action_size[0], node, hidden_n, key=key_c2),
        )
        critic_params_tree, critic_static = eqx.partition(critic_model, eqx.is_array)
        critic_params = _package_params(critic_params_tree)
        critic_fn = get_apply_fn_equinox_module(critic_static)

        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("actor", actor_params_tree)
                print_param("critic", critic_params_tree)
                print("---------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, actor_params, critic_params
        return preproc_fn, actor_fn, critic_fn

    return model_builder
