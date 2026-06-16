"""Golden-master param-tree structure for the shared flax DDPG/TD3 builders.

Locks the Flax param-tree keypaths + shapes that ``ddpg_builder``/``td3_builder``
produce after their deterministic ``Actor``/``Critic`` were extracted into the
shared ``ddpg_td3_blocks`` module. Flax param keys depend on the enclosing module
attribute names (``act``, ``crit1``, ``crit2``) and the inner class identifiers; a
future rename that silently broke checkpoint compatibility would change this
structure and fail here.
"""

from __future__ import annotations

import jax

from model_builder.flax.dpg.ddpg_builder import (
    model_builder_maker as ddpg_model_builder_maker,
)
from model_builder.flax.dpg.td3_builder import (
    model_builder_maker as td3_model_builder_maker,
)

_POLICY_KWARGS = {"node": 16, "hidden_n": 2, "embedding_mode": "normal"}
_OBSERVATION_SPACE = [[4]]
_ACTION_SIZE = [2]

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


def _param_structure(tree):
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {(jax.tree_util.keystr(path), tuple(leaf.shape)) for path, leaf in flat}


def _build(maker):
    builder = maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _preproc, _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    return policy_params, critic_params


def test_ddpg_builder_param_tree_structure():
    policy_params, critic_params = _build(ddpg_model_builder_maker)
    assert _param_structure(policy_params) == _ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _CRITIC_TOWER


def test_td3_builder_param_tree_structure():
    policy_params, critic_params = _build(td3_model_builder_maker)
    assert _param_structure(policy_params) == _ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _twin_tower("crit1") | _twin_tower("crit2")
