"""Golden-master param-tree structure for the Haiku DPG builders.

Locks the Haiku param-tree keypaths + shapes that the five Haiku DPG builders
(``ddpg``, ``td3``, ``sac``, ``tqc``, ``td7``) produce. Haiku param keys derive
from the ``hk.Module`` subclass name (lowercased, e.g. ``actor``/``critic``) and
the enclosing ``hk.transform`` scope, so extracting the shared ``Actor``/
``Critic`` classes into ``ddpg_td3_blocks`` must leave these structures
byte-identical. A future change that silently broke checkpoint compatibility
would change this structure and fail here.
"""

from __future__ import annotations

import jax

from model_builder.haiku.dpg.ddpg_builder import (
    model_builder_maker as ddpg_model_builder_maker,
)
from model_builder.haiku.dpg.sac_builder import (
    model_builder_maker as sac_model_builder_maker,
)
from model_builder.haiku.dpg.td3_builder import (
    model_builder_maker as td3_model_builder_maker,
)
from model_builder.haiku.dpg.td7_builder import (
    model_builder_maker as td7_model_builder_maker,
)
from model_builder.haiku.dpg.tqc_builder import (
    model_builder_maker as tqc_model_builder_maker,
)

_POLICY_KWARGS = {"node": 16, "hidden_n": 2, "embedding_mode": "normal"}
_OBSERVATION_SPACE = {"obs": [4]}
_ACTION_SIZE = [2]
_TQC_SUPPORT_N = 25


def _param_structure(tree):
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {(jax.tree_util.keystr(path), tuple(leaf.shape)) for path, leaf in flat}


# --- deterministic Actor (ddpg, td3): tanh head, output dim == action_size ---
_DET_ACTOR_STRUCTURE = {
    ("['actor/linear']['w']", (4, 16)),
    ("['actor/linear']['b']", (16,)),
    ("['actor/linear_1']['w']", (16, 16)),
    ("['actor/linear_1']['b']", (16,)),
    ("['actor/linear_2']['w']", (16, 2)),
    ("['actor/linear_2']['b']", (2,)),
}

# --- gaussian Actor (sac, tqc): output dim == action_size * 2 (mu, log_std) ---
_GAUSSIAN_ACTOR_STRUCTURE = {
    ("['actor/linear']['w']", (4, 16)),
    ("['actor/linear']['b']", (16,)),
    ("['actor/linear_1']['w']", (16, 16)),
    ("['actor/linear_1']['b']", (16,)),
    ("['actor/linear_2']['w']", (16, 4)),
    ("['actor/linear_2']['b']", (4,)),
}


def _critic_tower(name, head):
    return {
        (f"['{name}/linear']['w']", (6, 16)),
        (f"['{name}/linear']['b']", (16,)),
        (f"['{name}/linear_1']['w']", (16, 16)),
        (f"['{name}/linear_1']['b']", (16,)),
        (f"['{name}/linear_2']['w']", (16, head)),
        (f"['{name}/linear_2']['b']", (head,)),
    }


def test_ddpg_builder_param_tree_structure():
    builder = ddpg_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _preproc, _actor, _critic, params = builder(jax.random.PRNGKey(0))
    assert _param_structure(params) == _DET_ACTOR_STRUCTURE | _critic_tower("critic", 1)


def test_td3_builder_param_tree_structure():
    builder = td3_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _preproc, _actor, _critic, params = builder(jax.random.PRNGKey(0))
    assert _param_structure(params) == (
        _DET_ACTOR_STRUCTURE | _critic_tower("critic", 1) | _critic_tower("critic_1", 1)
    )


def test_sac_builder_param_tree_structure():
    builder = sac_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _preproc, _actor, _critic, params = builder(jax.random.PRNGKey(0))
    assert _param_structure(params) == (
        _GAUSSIAN_ACTOR_STRUCTURE | _critic_tower("critic", 1) | _critic_tower("critic_1", 1)
    )


def test_tqc_builder_param_tree_structure():
    builder = tqc_model_builder_maker(
        _OBSERVATION_SPACE, _ACTION_SIZE, _TQC_SUPPORT_N, dict(_POLICY_KWARGS)
    )
    _preproc, _actor, _critic, params = builder(jax.random.PRNGKey(0))
    assert _param_structure(params) == (
        _GAUSSIAN_ACTOR_STRUCTURE
        | _critic_tower("critic", _TQC_SUPPORT_N)
        | _critic_tower("critic_1", _TQC_SUPPORT_N)
    )


# --- td7 keeps its own Encoder/Action_Encoder/Actor/Critic (structurally
# distinct: avgl1norm preamble, embedding concat). These trees are locked here
# to prove the S2 extraction did not perturb td7 at all. ---
_TD7_ENCODER_PARAMS = {
    ("['encoder/linear']['w']", (4, 256)),
    ("['encoder/linear']['b']", (256,)),
    ("['encoder/linear_1']['w']", (256, 256)),
    ("['encoder/linear_1']['b']", (256,)),
    ("['encoder/linear_2']['w']", (256, 256)),
    ("['encoder/linear_2']['b']", (256,)),
    ("['action__encoder/linear']['w']", (258, 256)),
    ("['action__encoder/linear']['b']", (256,)),
    ("['action__encoder/linear_1']['w']", (256, 256)),
    ("['action__encoder/linear_1']['b']", (256,)),
    ("['action__encoder/linear_2']['w']", (256, 256)),
    ("['action__encoder/linear_2']['b']", (256,)),
}


def _td7_critic_tower(name):
    return {
        (f"['{name}/linear']['w']", (6, 16)),
        (f"['{name}/linear']['b']", (16,)),
        (f"['{name}/linear_1']['w']", (528, 16)),
        (f"['{name}/linear_1']['b']", (16,)),
        (f"['{name}/linear_2']['w']", (16, 16)),
        (f"['{name}/linear_2']['b']", (16,)),
        (f"['{name}/linear_3']['w']", (16, 1)),
        (f"['{name}/linear_3']['b']", (1,)),
    }


_TD7_ACTOR_PARAMS = {
    ("['actor/linear']['w']", (4, 16)),
    ("['actor/linear']['b']", (16,)),
    ("['actor/linear_1']['w']", (272, 16)),
    ("['actor/linear_1']['b']", (16,)),
    ("['actor/linear_2']['w']", (16, 16)),
    ("['actor/linear_2']['b']", (16,)),
    ("['actor/linear_3']['w']", (16, 2)),
    ("['actor/linear_3']['b']", (2,)),
}


def test_td7_builder_param_tree_structure():
    builder = td7_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    (
        _preproc,
        _encoder,
        _action_encoder,
        _actor,
        _critic,
        encoder_params,
        params,
    ) = builder(jax.random.PRNGKey(0))
    assert _param_structure(encoder_params) == _TD7_ENCODER_PARAMS
    assert _param_structure(params) == (
        _TD7_ACTOR_PARAMS | _td7_critic_tower("critic") | _td7_critic_tower("critic_1")
    )
