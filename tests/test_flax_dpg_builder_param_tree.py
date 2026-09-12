"""Independent Flax DPG parameter trees with different actor and critic widths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from model_builder.flax.dpg.ddpg_builder import (
    model_builder_maker as ddpg_model_builder_maker,
)
from model_builder.flax.dpg.sac_builder import (
    model_builder_maker as sac_model_builder_maker,
)
from model_builder.flax.dpg.td3_builder import (
    model_builder_maker as td3_model_builder_maker,
)
from model_builder.flax.dpg.tqc_builder import (
    model_builder_maker as tqc_model_builder_maker,
)
from model_builder.model_config import LayerConfig, MLPConfig, ResidualConfig

_POLICY_KWARGS = {
    "actor_model": MLPConfig((LayerConfig(16),) * 2),
    "critic_model": MLPConfig((LayerConfig(32),) * 2),
}
_OBSERVATION_SPACE = {"unified_obs": [4]}
_ACTION_SIZE = [2]
_SUPPORT_N = 25

_NETWORK_CONFIGURATIONS = [
    (
        _POLICY_KWARGS["actor_model"],
        _POLICY_KWARGS["critic_model"],
        {"Dense_0", "Dense_1", "Dense_2"},
    ),
    (
        ResidualConfig("simba", (16, 16)),
        ResidualConfig("simba", (32, 32)),
        {"Dense_0", "ResidualBlock_0", "ResidualBlock_1", "LayerNorm_0", "Dense_1"},
    ),
    (
        ResidualConfig("simbav2", (16, 16)),
        ResidualConfig("simbav2", (32, 32)),
        {"SimbaV2Embedding_0", "SimbaV2Block_0", "SimbaV2Block_1", "SimbaV2Head_0"},
    ),
    (
        _POLICY_KWARGS["actor_model"],
        ResidualConfig("simbav2", (32, 32)),
        {"SimbaV2Embedding_0", "SimbaV2Block_0", "SimbaV2Block_1", "SimbaV2Head_0"},
    ),
]

_ACTOR_STRUCTURE = {
    ("['params']['act']['Dense_0']['kernel']", (4, 16)),
    ("['params']['act']['Dense_0']['bias']", (16,)),
    ("['params']['act']['Dense_1']['kernel']", (16, 16)),
    ("['params']['act']['Dense_1']['bias']", (16,)),
    ("['params']['act']['Dense_2']['kernel']", (16, 2)),
    ("['params']['act']['Dense_2']['bias']", (2,)),
}

_CRITIC_TOWER = {
    ("['params']['Dense_0']['kernel']", (6, 32)),
    ("['params']['Dense_0']['bias']", (32,)),
    ("['params']['Dense_1']['kernel']", (32, 32)),
    ("['params']['Dense_1']['bias']", (32,)),
    ("['params']['Dense_2']['kernel']", (32, 1)),
    ("['params']['Dense_2']['bias']", (1,)),
}


def _twin_tower(name):
    return {
        (path.replace("['params']", f"['params']['{name}']", 1), shape)
        for path, shape in _CRITIC_TOWER
    }


# Squashed-Gaussian actor (gaussian_blocks.Actor): a shared two-layer MLP stack
# (Dense_0/Dense_1) followed by TWO heads -- the mu head (Dense_2) and the
# log_std head (Dense_3). The extra Dense_3 vs the deterministic ddpg/td3 actor is
# the load-bearing difference; dropping the log_std head would fail this set.
_GAUSSIAN_ACTOR_STRUCTURE = {
    ("['params']['act']['Dense_0']['kernel']", (4, 16)),
    ("['params']['act']['Dense_0']['bias']", (16,)),
    ("['params']['act']['Dense_1']['kernel']", (16, 16)),
    ("['params']['act']['Dense_1']['bias']", (16,)),
    ("['params']['act']['Dense_2']['kernel']", (16, 2)),
    ("['params']['act']['Dense_2']['bias']", (2,)),
    ("['params']['act']['Dense_3']['kernel']", (16, 2)),
    ("['params']['act']['Dense_3']['bias']", (2,)),
}


def _quantile_critic_tower(support_n):
    # tqc's Critic mirrors the plain Critic MLP but emits ``support_n`` quantiles
    # from its final Dense instead of a single scalar.
    return {
        ("['params']['Dense_0']['kernel']", (6, 32)),
        ("['params']['Dense_0']['bias']", (32,)),
        ("['params']['Dense_1']['kernel']", (32, 32)),
        ("['params']['Dense_1']['bias']", (32,)),
        ("['params']['Dense_2']['kernel']", (32, support_n)),
        ("['params']['Dense_2']['bias']", (support_n,)),
    }


def _twin_tower_from(name, tower):
    return {
        (path.replace("['params']", f"['params']['{name}']", 1), shape) for path, shape in tower
    }


def _param_structure(tree):
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {(jax.tree_util.keystr(path), tuple(leaf.shape)) for path, leaf in flat}


def _build(maker):
    builder = maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    return policy_params, critic_params


@pytest.mark.parametrize(
    "make_builder, twin", [(ddpg_model_builder_maker, False), (td3_model_builder_maker, True)]
)
@pytest.mark.parametrize("actor_model, critic_model, critic_roots", _NETWORK_CONFIGURATIONS)
def test_deterministic_builders_preserve_public_contract_and_param_roots(
    make_builder, twin, actor_model, critic_model, critic_roots
):
    builder = make_builder(
        _OBSERVATION_SPACE,
        _ACTION_SIZE,
        {**_POLICY_KWARGS, "actor_model": actor_model, "critic_model": critic_model},
    )
    assert len(builder()) == 2
    actor, critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert set(policy_params["params"]) == {"act"}
    assert set(critic_params["params"]) == ({"crit1", "crit2"} if twin else {"crit1"})
    for critic_params_tree in critic_params["params"].values():
        assert set(critic_params_tree) == critic_roots

    key = jax.random.PRNGKey(1)
    observations = {"unified_obs": jnp.zeros((1, 4), dtype=jnp.float32)}
    action = actor(policy_params, key, observations)
    values = critic(critic_params, policy_params, key, observations, action)
    assert action.shape == (1, 2)
    if twin:
        assert tuple(value.shape for value in values) == ((1, 1), (1, 1))
    else:
        assert values.shape == (1, 1)


def test_ddpg_builder_param_tree_structure():
    policy_params, critic_params = _build(ddpg_model_builder_maker)
    assert _param_structure(policy_params) == _ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower("crit1")


def test_td3_builder_param_tree_structure():
    policy_params, critic_params = _build(td3_model_builder_maker)
    assert _param_structure(policy_params) == _ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower("crit1") | _twin_tower("crit2")


def test_sac_builder_param_tree_structure():
    policy_params, critic_params = _build(sac_model_builder_maker)
    assert _param_structure(policy_params) == _GAUSSIAN_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower("crit1") | _twin_tower("crit2")


def test_tqc_builder_param_tree_structure():
    builder = tqc_model_builder_maker(
        _OBSERVATION_SPACE, _ACTION_SIZE, _SUPPORT_N, dict(_POLICY_KWARGS)
    )
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    quantile_tower = _quantile_critic_tower(_SUPPORT_N)
    assert _param_structure(policy_params) == _GAUSSIAN_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower_from(
        "crit1", quantile_tower
    ) | _twin_tower_from("crit2", quantile_tower)
