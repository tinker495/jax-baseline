"""Regression tests for the ``scaled_by_reset`` soft-reset path in the DPG family.

Locks two bugs that made ``scaled_by_reset=True`` (a non-default periodic soft
reset of params + optimizer state) crash or silently corrupt state:

Bug 1 (all six algos: DDPG/TD3/SAC/TQC/CrossQ/TD7) -- the call sites assigned the
2-tuple ``(tensors, optimizer_state)`` returned by ``scaled_by_reset`` to a single
variable, so ``policy_params``/``critic_params`` became a 2-tuple instead of a
pytree (and the reinitialized optimizer state was discarded). The next
``optax.apply_updates`` / return-structure consumer then operated on a tuple.

Bug 2 (DDPG only) -- ``_train_on_batch`` / ``_train_on_bulk`` threaded ``None`` as
the PRNG key into ``_train_step``; the ``scaled_by_reset`` branch needs a real key
to draw the reset noise, so it crashed under ``jax.random.split``/``normal``.

Both run on CPU with ``reset_freq=1`` so the reset branch fires on step 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.TD3.td3 import TD3
from model_builder.flax.dpg.ddpg_builder import (
    model_builder_maker as ddpg_model_builder_maker,
)
from model_builder.flax.dpg.td3_builder import (
    model_builder_maker as td3_model_builder_maker,
)

_POLICY_KWARGS = {"node": 16, "hidden_n": 2, "embedding_mode": "normal"}
_OBSERVATION_SPACE = [[4]]
_ACTION_SIZE = [2]
_BATCH_SIZE = 8


def _batch():
    return {
        "obses": [np.ones((_BATCH_SIZE, 4), dtype=np.float32)],
        "actions": np.zeros((_BATCH_SIZE, _ACTION_SIZE[0]), dtype=np.float32),
        "rewards": np.ones((_BATCH_SIZE, 1), dtype=np.float32),
        "nxtobses": [np.full((_BATCH_SIZE, 4), 2.0, dtype=np.float32)],
        "terminateds": np.zeros((_BATCH_SIZE, 1), dtype=np.float32),
    }


def _make_ddpg():
    agent = DDPG.__new__(DDPG)
    builder = ddpg_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    agent.preproc, agent.actor, agent.critic, agent.policy_params, agent.critic_params = builder(
        jax.random.PRNGKey(0)
    )
    agent.optimizer = optax.adam(1e-3)
    agent.opt_policy_state = agent.optimizer.init(agent.policy_params)
    agent.opt_critic_state = agent.optimizer.init(agent.critic_params)
    agent.target_policy_params = agent.policy_params
    agent.target_critic_params = agent.critic_params
    agent.target_network_update_tau = 0.005
    agent._gamma = 0.99
    agent.prioritized_replay = False
    agent.scaled_by_reset = True
    agent.reset_freq = 1
    return agent


def _make_td3():
    agent = TD3.__new__(TD3)
    builder = td3_model_builder_maker(_OBSERVATION_SPACE, _ACTION_SIZE, dict(_POLICY_KWARGS))
    agent.preproc, agent.actor, agent.critic, agent.policy_params, agent.critic_params = builder(
        jax.random.PRNGKey(0)
    )
    agent.optimizer = optax.adam(1e-3)
    agent.opt_policy_state = agent.optimizer.init(agent.policy_params)
    agent.opt_critic_state = agent.optimizer.init(agent.critic_params)
    agent.target_policy_params = agent.policy_params
    agent.target_critic_params = agent.critic_params
    agent.target_network_update_tau = 0.005
    agent._gamma = 0.99
    agent.prioritized_replay = False
    agent.scaled_by_reset = True
    agent.reset_freq = 1
    agent.policy_delay = 2
    agent.action_noise = 0.1
    agent.target_action_noise = 0.2
    agent.action_noise_clamp = 0.5
    agent.batch_size = _BATCH_SIZE
    agent.action_size = _ACTION_SIZE
    return agent


def _assert_valid_params(new_params, reference):
    # Bug 1 regression: a single-variable assign of scaled_by_reset's 2-tuple
    # would make these a (params, opt_state) tuple, not a pytree matching the
    # original structure.
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(reference)
    assert not isinstance(new_params, tuple)


def test_ddpg_scaled_by_reset_runs_and_returns_valid_pytrees():
    # DDPG = both bugs (key=None + tuple-assign).
    agent = _make_ddpg()
    out = agent._train_step(
        agent.policy_params,
        agent.critic_params,
        agent.target_policy_params,
        agent.target_critic_params,
        agent.opt_policy_state,
        agent.opt_critic_state,
        jnp.asarray(0),
        jax.random.PRNGKey(1),
        **_batch(),
    )
    policy_params, critic_params = out[0], out[1]
    _assert_valid_params(policy_params, agent.policy_params)
    _assert_valid_params(critic_params, agent.critic_params)


def test_td3_scaled_by_reset_runs_and_returns_valid_pytrees():
    # TD3 = tuple-assign bug only (already threads a real key).
    agent = _make_td3()
    out = agent._train_step(
        agent.policy_params,
        agent.critic_params,
        agent.target_policy_params,
        agent.target_critic_params,
        agent.opt_policy_state,
        agent.opt_critic_state,
        jax.random.PRNGKey(1),
        jnp.asarray(0),
        **_batch(),
    )
    policy_params, critic_params = out[0], out[1]
    _assert_valid_params(policy_params, agent.policy_params)
    _assert_valid_params(critic_params, agent.critic_params)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
