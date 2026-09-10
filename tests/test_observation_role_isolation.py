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
    kwargs = {"node": 16, "hidden_n": 1}
    key = jax.random.PRNGKey(3)
    if family == "ac":
        preproc, actor, critic, params = module.model_builder_maker(
            space, (2,), "continuous", kwargs
        )(key)
        critic_params = params
    elif backend == "flax":
        preproc, actor, critic, params, critic_params = module.model_builder_maker(
            space, (2,), kwargs
        )(key)
    else:
        preproc, actor, critic, params = module.model_builder_maker(space, (2,), kwargs)(key)
        critic_params = params

    obs = {name: jnp.ones((2, *shape)) for name, shape in space.items()}

    def outputs(observations):
        feature = preproc(params, key, observations)
        policy = actor(params, key, feature)
        value = (
            critic(critic_params, key, feature)
            if family == "ac"
            else critic(critic_params, key, feature, jnp.zeros((2, 2)))
        )
        return policy[0] if family == "ac" else policy, value

    features = preproc(params, key, obs)
    assert features["actor"].shape == (2, 3)
    assert features["critic"].shape == (2, 4)
    for role, name in (("actor", "actor_sensor"), ("critic", "critic_privileged")):
        changed = preproc(params, key, {**obs, name: obs[name] * 7})
        assert not np.array_equal(changed[role], features[role])
    shared = preproc(params, key, {**obs, "unified_command": obs["unified_command"] * 7})
    for role in ("actor", "critic"):
        assert not np.array_equal(shared[role], features[role])

    policy, value = outputs(obs)
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
    built = module.model_builder_maker(space, (2,), {"node": 16, "hidden_n": 1})(key)
    preproc, encoder, _, actor, _ = built[:5]
    encoder_params, policy_params = built[5:7]
    obs = {name: jnp.ones((2, *shape)) for name, shape in space.items()}
    feature = preproc(encoder_params, key, obs)
    privileged = preproc(
        encoder_params, key, {**obs, "critic_privileged": obs["critic_privileged"] * 7}
    )
    zs = encoder(encoder_params, key, feature)
    privileged_zs = encoder(encoder_params, key, privileged)
    np.testing.assert_array_equal(privileged_zs, zs)
    np.testing.assert_array_equal(
        actor(policy_params, key, feature, zs),
        actor(policy_params, key, privileged, privileged_zs),
    )
