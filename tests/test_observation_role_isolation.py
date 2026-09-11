"""Policy outputs must not leak privileged critic-only observations."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["flax", "haiku"])
@pytest.mark.parametrize("family", ["ac", "dpg"])
def test_actor_critic_observation_roles_are_isolated(backend, family):
    module = importlib.import_module(
        f"model_builder.{backend}.{family}.{'ac' if family == 'ac' else 'ddpg'}_builder"
    )
    space = {"actor_sensor": [2], "critic_privileged": [3], "unified_command": [1]}
    kwargs = {"actor_node": 16, "critic_node": 32, "hidden_n": 1}
    key = jax.random.PRNGKey(3)
    if family == "ac":
        actor, critic, params, critic_params = module.model_builder_maker(
            space, (2,), "continuous", kwargs
        )(key)
    else:
        actor, critic, params, critic_params = module.model_builder_maker(space, (2,), kwargs)(key)

    obs = {name: jnp.ones((2, *shape)) for name, shape in space.items()}

    def outputs(observations):
        policy = actor(params, key, observations)
        value = (
            critic(critic_params, key, observations)
            if family == "ac"
            else critic(critic_params, key, observations, jnp.zeros((2, 2)))
        )
        return policy[0] if family == "ac" else policy, value

    policy, value = outputs(obs)
    actor_only = actor(
        params, key, {name: obs[name] for name in ("actor_sensor", "unified_command")}
    )
    np.testing.assert_array_equal(actor_only[0] if family == "ac" else actor_only, policy)
    actor_changed, critic_unchanged = outputs({**obs, "actor_sensor": obs["actor_sensor"] * 7})
    actor_unchanged, _ = outputs({**obs, "critic_privileged": obs["critic_privileged"] * 7})
    shared_actor, _ = outputs({**obs, "unified_command": obs["unified_command"] * 7})
    np.testing.assert_array_equal(actor_unchanged, policy)
    np.testing.assert_array_equal(critic_unchanged, value)
    assert not np.allclose(actor_changed, policy)
    assert not np.allclose(shared_actor, policy)


@pytest.mark.parametrize("backend", ["flax", "haiku"])
def test_td7_encoder_and_actor_ignore_privileged_observations(backend):
    module = importlib.import_module(f"model_builder.{backend}.dpg.td7_builder")
    space = {"actor_sensor": [2], "critic_privileged": [3], "unified_command": [1]}
    key = jax.random.PRNGKey(3)
    built = module.model_builder_maker(
        space, (2,), {"actor_node": 16, "critic_node": 32, "hidden_n": 1}
    )(key)
    encoder, _, _, _, actor, _ = built[:6]
    encoder_params, _, policy_params, _ = built[6:]
    obs = {name: jnp.ones((2, *shape)) for name, shape in space.items()}
    feature, zs = encoder(encoder_params, key, obs)
    privileged, privileged_zs = encoder(
        encoder_params, key, {**obs, "critic_privileged": obs["critic_privileged"] * 7}
    )
    actor_only, actor_only_zs = encoder(
        encoder_params, key, {name: obs[name] for name in ("actor_sensor", "unified_command")}
    )
    np.testing.assert_array_equal(actor_only, feature)
    np.testing.assert_array_equal(actor_only_zs, zs)
    np.testing.assert_array_equal(privileged_zs, zs)
    np.testing.assert_array_equal(
        actor(policy_params, key, feature, zs),
        actor(policy_params, key, privileged, privileged_zs),
    )
