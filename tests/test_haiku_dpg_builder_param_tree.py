"""Parameter shapes for independently owned Haiku DPG actor and critic models."""

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
from model_builder.model_config import LayerConfig, MLPConfig

_POLICY_KWARGS = {
    "actor_model": MLPConfig((LayerConfig(16),) * 2),
    "critic_model": MLPConfig((LayerConfig(16),) * 2),
}
_OBSERVATION_SPACE = {"unified_obs": [4]}
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
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert _param_structure(policy_params) == _DET_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _critic_tower("critic", 1)


def test_td3_builder_param_tree_structure():
    builder = td3_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert _param_structure(policy_params) == _DET_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _critic_tower("critic", 1) | _critic_tower(
        "critic_1", 1
    )


def test_sac_builder_param_tree_structure():
    builder = sac_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert _param_structure(policy_params) == _GAUSSIAN_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == _critic_tower("critic", 1) | _critic_tower(
        "critic_1", 1
    )


def test_tqc_builder_param_tree_structure():
    builder = tqc_model_builder_maker(
        _OBSERVATION_SPACE, _ACTION_SIZE, _TQC_SUPPORT_N, dict(_POLICY_KWARGS)
    )
    _actor, _critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))
    assert _param_structure(policy_params) == _GAUSSIAN_ACTOR_STRUCTURE
    assert _param_structure(critic_params) == (
        _critic_tower("critic", _TQC_SUPPORT_N) | _critic_tower("critic_1", _TQC_SUPPORT_N)
    )


# TD7 gives each role its own state and action encoders.
_TD7_ENCODER_PARAMS = {
    ("['encoder/linear']['w']", (4, 16)),
    ("['encoder/linear']['b']", (16,)),
    ("['encoder/linear_1']['w']", (16, 16)),
    ("['encoder/linear_1']['b']", (16,)),
    ("['encoder/linear_2']['w']", (16, 16)),
    ("['encoder/linear_2']['b']", (16,)),
    ("['action__encoder/linear']['w']", (18, 16)),
    ("['action__encoder/linear']['b']", (16,)),
    ("['action__encoder/linear_1']['w']", (16, 16)),
    ("['action__encoder/linear_1']['b']", (16,)),
    ("['action__encoder/linear_2']['w']", (16, 16)),
    ("['action__encoder/linear_2']['b']", (16,)),
}


def _td7_critic_tower(name):
    return {
        (f"['{name}/linear']['w']", (6, 16)),
        (f"['{name}/linear']['b']", (16,)),
        (f"['{name}/linear_1']['w']", (48, 16)),
        (f"['{name}/linear_1']['b']", (16,)),
        (f"['{name}/linear_2']['w']", (16, 16)),
        (f"['{name}/linear_2']['b']", (16,)),
        (f"['{name}/linear_3']['w']", (16, 1)),
        (f"['{name}/linear_3']['b']", (1,)),
    }


_TD7_ACTOR_PARAMS = {
    ("['actor/linear']['w']", (4, 16)),
    ("['actor/linear']['b']", (16,)),
    ("['actor/linear_1']['w']", (32, 16)),
    ("['actor/linear_1']['b']", (16,)),
    ("['actor/linear_2']['w']", (16, 16)),
    ("['actor/linear_2']['b']", (16,)),
    ("['actor/linear_3']['w']", (16, 2)),
    ("['actor/linear_3']['b']", (2,)),
}


def test_td7_builder_param_tree_structure():
    builder = td7_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    (
        _actor_encoder,
        _critic_encoder,
        _actor_action_encoder,
        _critic_action_encoder,
        _actor,
        _critic,
        actor_encoder_params,
        critic_encoder_params,
        policy_params,
        critic_params,
    ) = builder(jax.random.PRNGKey(0))
    assert _param_structure(actor_encoder_params) == _TD7_ENCODER_PARAMS
    assert _param_structure(critic_encoder_params) == _TD7_ENCODER_PARAMS
    assert _param_structure(policy_params) == _TD7_ACTOR_PARAMS
    assert _param_structure(critic_params) == _td7_critic_tower("critic") | _td7_critic_tower(
        "critic_1"
    )
