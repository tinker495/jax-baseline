"""Regression tests for Munchausen quantile TD targets.

The Munchausen paper's M-IQN target uses the policy expectation over next actions:
``sum_a pi(a|s') * (z(s', a) - tau log pi(a|s'))``. These tests pin that expectation
directly so a sampled-action target cannot slip back in.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.FQF.fqf import FQF
from jax_baselines.IQN.iqn import IQN
from jax_baselines.math.policy_math import q_log_pi
from jax_baselines.QRDQN.qrdqn import QRDQN

BATCH = 2
ACTIONS = 3
SUPPORT = 4
ALPHA = 0.4
TAU = 0.7
GAMMA = 0.83
KEY = jax.random.PRNGKey(7)

TARGET_QUANTILES = jnp.asarray(
    [
        [[0.0, 0.1, 0.2, 0.3], [1.2, 1.3, 1.4, 1.5], [2.4, 2.5, 2.6, 2.7]],
        [[-0.5, -0.4, -0.3, -0.2], [0.6, 0.7, 0.8, 0.9], [1.8, 1.9, 2.0, 2.1]],
    ],
    dtype=jnp.float32,
)
ONLINE_QUANTILES = jnp.asarray(
    [
        [[0.3, 0.4, 0.5, 0.6], [0.9, 1.0, 1.1, 1.2], [1.4, 1.5, 1.6, 1.7]],
        [[1.0, 1.1, 1.2, 1.3], [0.3, 0.4, 0.5, 0.6], [-0.2, -0.1, 0.0, 0.1]],
    ],
    dtype=jnp.float32,
)
REWARDS = jnp.asarray([[0.25], [-0.5]], dtype=jnp.float32)
NOT_TERMINATED = jnp.asarray([[1.0], [0.5]], dtype=jnp.float32)
ACTIONS_TAKEN = jnp.asarray([[[0]], [[2]]], dtype=jnp.int32)
OBSES = jnp.zeros((BATCH, 1), dtype=jnp.float32)


def _fake_agent(get_q, double_q):
    return SimpleNamespace(
        get_q=get_q,
        munchausen=True,
        double_q=double_q,
        munchausen_entropy_tau=TAU,
        munchausen_alpha=ALPHA,
        _gamma=GAMMA,
        batch_size=BATCH,
        n_support=SUPPORT,
    )


def _expected_mean_quantile_target(double_q):
    selection_quantiles = ONLINE_QUANTILES if double_q else TARGET_QUANTILES
    selection_q = jnp.mean(selection_quantiles, axis=2)
    next_sub_q, tau_log_pi_next = q_log_pi(selection_q, TAU)
    pi_next = jnp.expand_dims(jax.nn.softmax(next_sub_q / TAU), axis=2)
    next_vals = jnp.sum(
        pi_next * (TARGET_QUANTILES - jnp.expand_dims(tau_log_pi_next, axis=2)), axis=1
    )
    next_vals = NOT_TERMINATED * next_vals

    behavior_quantiles = ONLINE_QUANTILES if double_q else TARGET_QUANTILES
    behavior_q = jnp.mean(behavior_quantiles, axis=2)
    _, tau_log_pi = q_log_pi(behavior_q, TAU, clip=True)
    addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(ACTIONS_TAKEN, axis=2), axis=1)
    return GAMMA * next_vals + REWARDS + ALPHA * addon


@pytest.mark.parametrize("double_q", [False, True])
def test_qrdqn_munchausen_target_uses_policy_expectation(double_q):
    def get_q(params, obses, key):
        return params

    out = QRDQN._target(
        _fake_agent(get_q, double_q),
        ONLINE_QUANTILES,
        TARGET_QUANTILES,
        OBSES,
        ACTIONS_TAKEN,
        REWARDS,
        OBSES,
        NOT_TERMINATED,
        KEY,
    )

    np.testing.assert_allclose(
        np.asarray(out), np.asarray(_expected_mean_quantile_target(double_q)), atol=1e-6
    )


@pytest.mark.parametrize("double_q", [False, True])
def test_iqn_munchausen_target_uses_policy_expectation(double_q):
    def get_q(params, obses, tau, key):
        return params

    out = IQN._target(
        _fake_agent(get_q, double_q),
        ONLINE_QUANTILES,
        TARGET_QUANTILES,
        OBSES,
        ACTIONS_TAKEN,
        REWARDS,
        OBSES,
        NOT_TERMINATED,
        KEY,
    )

    np.testing.assert_allclose(
        np.asarray(out), np.asarray(_expected_mean_quantile_target(double_q)), atol=1e-6
    )


FQF_TAUS = jnp.asarray([[0.0, 0.1, 0.4, 0.75, 1.0], [0.0, 0.2, 0.45, 0.8, 1.0]])
FQF_TAU_HATS = (FQF_TAUS[:, :-1] + FQF_TAUS[:, 1:]) / 2.0


def _fqf_weighted_q(quantiles, tau):
    weights = tau[:, 1:] - tau[:, :-1]
    return jnp.sum(jnp.expand_dims(weights, axis=1) * quantiles, axis=2)


def _expected_fqf_target(double_q):
    weights = FQF_TAUS[:, 1:] - FQF_TAUS[:, :-1]
    selection_quantiles = ONLINE_QUANTILES if double_q else TARGET_QUANTILES
    selection_q = _fqf_weighted_q(selection_quantiles, FQF_TAUS)
    next_sub_q, tau_log_pi_next = q_log_pi(selection_q, TAU)
    pi_next = jnp.expand_dims(jax.nn.softmax(next_sub_q / TAU), axis=2)
    next_vals = jnp.sum(
        pi_next * (TARGET_QUANTILES - jnp.expand_dims(tau_log_pi_next, axis=2)), axis=1
    )
    next_vals = NOT_TERMINATED * next_vals

    behavior_quantiles = ONLINE_QUANTILES if double_q else TARGET_QUANTILES
    behavior_q = _fqf_weighted_q(behavior_quantiles, FQF_TAUS)
    _, tau_log_pi = q_log_pi(behavior_q, TAU, clip=True)
    addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(ACTIONS_TAKEN, axis=2), axis=1)
    return GAMMA * next_vals + REWARDS + ALPHA * addon, weights


@pytest.mark.parametrize("double_q", [False, True])
def test_fqf_munchausen_target_uses_policy_expectation(double_q):
    obj = SimpleNamespace(
        preproc=lambda params, key, obses: obses,
        fpf=lambda fqf_params, key, feature: (FQF_TAUS, FQF_TAU_HATS, None),
        get_quantile=lambda params, feature, tau_hat, key: params,
        get_q=lambda params, feature, tau, tau_hat, key: _fqf_weighted_q(params, tau),
        quantiles_to_q=_fqf_weighted_q,
        munchausen=True,
        double_q=double_q,
        munchausen_entropy_tau=TAU,
        munchausen_alpha=ALPHA,
        _gamma=GAMMA,
        n_support=SUPPORT,
    )

    out, out_weights = FQF._target(
        obj,
        ONLINE_QUANTILES,
        None,
        TARGET_QUANTILES,
        FQF_TAUS,
        FQF_TAU_HATS,
        OBSES,
        ACTIONS_TAKEN,
        REWARDS,
        OBSES,
        NOT_TERMINATED,
        KEY,
    )
    expected, expected_weights = _expected_fqf_target(double_q)

    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_weights), np.asarray(expected_weights), atol=1e-6)
