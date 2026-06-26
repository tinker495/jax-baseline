"""Canonical-equivalence regression guard for the APE-X Munchausen `_target` fork.

CONTEXT.md ("Canonical-semantics decisions") declares the canonical Munchausen target
to be the local (non-distributed) form: the addon is the clipped ``tau_log_pi[actions]``
returned by ``q_log_pi``, and ``q_k_targets`` is sourced from the online params under
double-Q. The APE-X scalar/quantile variants (apex_dqn / apex_qrdqn / apex_iqn) had
forked onto ``log_pi = q_sub - tau * tau_log_pi`` with no double-Q branch, exactly the
fork that ``apex_c51`` already converged away from on 2026-06-05.

These tests pin the fix: each APE-X ``_target`` must produce the *same* tensor as its
canonical sibling's ``_target`` on identical inputs. Before the convergence they FAIL
(proving the divergence); after it they PASS (proving canonical equivalence) and stand
as a permanent guard against re-forking.

The methods are exercised in isolation via a lightweight fake carrying a deterministic
``get_q`` plus the handful of scalar attributes ``_target`` reads -- no Ray, no model
build, no replay buffer.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.DQN.apex_dqn import APE_X_DQN
from jax_baselines.DQN.dqn import DQN
from jax_baselines.IQN.apex_iqn import APE_X_IQN
from jax_baselines.IQN.iqn import IQN
from jax_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN
from jax_baselines.QRDQN.qrdqn import QRDQN

BATCH = 4
ACTIONS = 3
FEAT = 5
N_SUPPORT = 8
TAU = 0.03
ALPHA = 0.9
GAMMA = 0.99
SUPPORT_RAMP = jnp.linspace(-0.5, 0.5, N_SUPPORT)


def _fake(get_q, munchausen, double_q):
    return SimpleNamespace(
        get_q=get_q,
        munchausen=munchausen,
        double_q=double_q,
        munchausen_entropy_tau=TAU,
        munchausen_alpha=ALPHA,
        _gamma=GAMMA,
        batch_size=BATCH,
        n_support=N_SUPPORT,
    )


def _common_inputs(action_shape):
    rng = np.random.default_rng(0)
    obses = jnp.asarray(rng.standard_normal((BATCH, FEAT)), dtype=jnp.float32)
    nxtobses = jnp.asarray(rng.standard_normal((BATCH, FEAT)), dtype=jnp.float32)
    params = jnp.asarray(rng.standard_normal((FEAT, ACTIONS)), dtype=jnp.float32)
    target_params = jnp.asarray(rng.standard_normal((FEAT, ACTIONS)), dtype=jnp.float32)
    actions = jnp.asarray(rng.integers(0, ACTIONS, size=action_shape), dtype=jnp.int32)
    rewards = jnp.asarray(rng.standard_normal((BATCH, 1)), dtype=jnp.float32)
    not_terminateds = jnp.asarray(rng.integers(0, 2, size=(BATCH, 1)), dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    return (
        params,
        target_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    )


def _assert_equiv(apex_cls, canon_cls, get_q, action_shape, munchausen, double_q):
    obj = _fake(get_q, munchausen, double_q)
    inputs = _common_inputs(action_shape)
    out_apex = apex_cls._target(obj, *inputs)
    out_canon = canon_cls._target(obj, *inputs)
    np.testing.assert_allclose(np.asarray(out_apex), np.asarray(out_canon), atol=1e-5, rtol=1e-5)


CASES = [(True, False), (True, True), (False, False), (False, True)]


@pytest.mark.parametrize("munchausen,double_q", CASES)
def test_apex_dqn_target_matches_canonical(munchausen, double_q):
    def get_q(params, obs, key):
        return jnp.tanh(obs @ params)  # (B, A)

    _assert_equiv(APE_X_DQN, DQN, get_q, (BATCH, 1), munchausen, double_q)


@pytest.mark.parametrize("munchausen,double_q", CASES)
def test_apex_qrdqn_target_matches_canonical(munchausen, double_q):
    def get_q(params, obs, key):
        base = jnp.tanh(obs @ params)  # (B, A)
        return base[:, :, None] + SUPPORT_RAMP  # (B, A, S)

    _assert_equiv(APE_X_QRDQN, QRDQN, get_q, (BATCH, 1, 1), munchausen, double_q)


@pytest.mark.parametrize("munchausen,double_q", CASES)
def test_apex_iqn_target_matches_canonical(munchausen, double_q):
    def get_q(params, obs, tau, key):
        base = jnp.tanh(obs @ params)  # (B, A)
        return base[:, :, None] + 0.1 * tau[:, None, :]  # (B, A, S)

    _assert_equiv(APE_X_IQN, IQN, get_q, (BATCH, 1, 1), munchausen, double_q)
