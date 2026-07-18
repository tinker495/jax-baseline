"""Golden-master param-tree structure for the shared flax DPG builders.

Locks the Flax param-tree keypaths + shapes that ``ddpg_builder``/``td3_builder``
produce after their deterministic ``Actor``/``Critic`` were extracted into the
shared ``ddpg_td3_blocks`` module, and that ``sac_builder``/``tqc_builder``
produce after their squashed-Gaussian ``Actor`` + plain ``Critic`` were extracted
into the shared ``gaussian_blocks`` module. Flax param keys depend on the
enclosing module attribute names (``act``, ``crit1``, ``crit2``) and the inner
class identifiers; a future rename or layer-count change that silently broke
checkpoint compatibility would change this structure and fail here.
"""

from __future__ import annotations

import importlib

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

_POLICY_KWARGS = {"node": 16, "hidden_n": 2, "embedding_mode": "normal"}
_OBSERVATION_SPACE = {"obs": [4]}
_ACTION_SIZE = [2]
_SUPPORT_N = 25

_DETERMINISTIC_BUILDERS = [
    ("ddpg_builder", {"Dense_0", "Dense_1", "Dense_2"}, False),
    (
        "simba_ddpg_builder",
        {"Dense_0", "ResidualBlock_0", "ResidualBlock_1", "LayerNorm_0", "Dense_1"},
        False,
    ),
    (
        "simbav2_ddpg_builder",
        {"SimbaV2Embedding_0", "SimbaV2Block_0", "SimbaV2Block_1", "SimbaV2Head_0"},
        False,
    ),
    ("td3_builder", {"crit1", "crit2"}, True),
    ("simba_td3_builder", {"crit1", "crit2"}, True),
    ("simbav2_td3_builder", {"crit1", "crit2"}, True),
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
    ("['params']['Dense_0']['kernel']", (6, 16)),
    ("['params']['Dense_0']['bias']", (16,)),
    ("['params']['Dense_1']['kernel']", (16, 16)),
    ("['params']['Dense_1']['bias']", (16,)),
    ("['params']['Dense_2']['kernel']", (16, 1)),
    ("['params']['Dense_2']['bias']", (1,)),
}


def _twin_tower(name):
    return {
        (path.replace("['params']", f"['params']['{name}']", 1), shape)
        for path, shape in _CRITIC_TOWER
    }


# Squashed-Gaussian actor (gaussian_blocks.Actor): a shared 2*hidden_n MLP stack
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
        ("['params']['Dense_0']['kernel']", (6, 16)),
        ("['params']['Dense_0']['bias']", (16,)),
        ("['params']['Dense_1']['kernel']", (16, 16)),
        ("['params']['Dense_1']['bias']", (16,)),
        ("['params']['Dense_2']['kernel']", (16, support_n)),
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
    _preproc, _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    return policy_params, critic_params


@pytest.mark.parametrize("module_name, critic_roots, twin", _DETERMINISTIC_BUILDERS)
def test_deterministic_builders_preserve_public_contract_and_param_roots(
    module_name, critic_roots, twin
):
    module = importlib.import_module(f"model_builder.flax.dpg.{module_name}")
    assert hasattr(module, "Actor")
    assert hasattr(module, "Critic")

    builder = module.model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    assert len(builder()) == 3
    preproc, actor, critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert set(policy_params["params"]) == {"act"}
    assert set(critic_params["params"]) == critic_roots

    key = jax.random.PRNGKey(1)
    feature = preproc(policy_params, key, {"obs": jnp.zeros((1, 4), dtype=jnp.float32)})
    action = actor(policy_params, key, feature)
    values = critic(critic_params, key, feature, action)
    assert feature.shape == (1, 4)
    assert action.shape == (1, 2)
    if twin:
        assert tuple(value.shape for value in values) == ((1, 1), (1, 1))
    else:
        assert values.shape == (1, 1)


def test_ddpg_builder_param_tree_structure():
    policy_params, critic_params = _build(ddpg_model_builder_maker)
    assert _param_structure(policy_params) == _ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _CRITIC_TOWER


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
    _preproc, _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    quantile_tower = _quantile_critic_tower(_SUPPORT_N)
    assert _param_structure(policy_params) == _GAUSSIAN_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower_from(
        "crit1", quantile_tower
    ) | _twin_tower_from("crit2", quantile_tower)
