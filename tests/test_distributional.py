"""Pure / characterization tests for jax_baselines.common.distributional.

These lock the canonical distributional-TD-target semantics (``c51.py`` /
``hl_gauss_c51.py``) into a representation-agnostic operator, using synthetic
arrays only — no network calls, no PRNG. The characterization tests inline the
canonical inline math and assert the shared operator reproduces it bit-close.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.common.distributional import (
    CategoricalBackend,
    HLGaussBackend,
    HLGaussTransform,
    MunchausenSpec,
    categorical_projection,
    distributional_td_target,
)
from jax_baselines.common.losses import QuantileHuberLosses
from jax_baselines.common.policy_math import q_log_pi

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# helpers: a small categorical support
# ---------------------------------------------------------------------------
def make_support(vmin=-5.0, vmax=5.0, n_bins=11):
    support = np.linspace(vmin, vmax, n_bins, dtype=np.float32)
    delta = float((vmax - vmin) / (n_bins - 1))
    return support, vmin, vmax, delta, n_bins


def make_cat_backend(vmin=-5.0, vmax=5.0, n_bins=11):
    support, vmin, vmax, delta, n_bins = make_support(vmin, vmax, n_bins)
    return CategoricalBackend(
        support=jnp.asarray(support),
        support_min=vmin,
        support_max=vmax,
        delta=delta,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# categorical_projection
# ---------------------------------------------------------------------------
def test_projection_rows_sum_to_one():
    support, vmin, vmax, delta, n_bins = make_support()
    rng = np.random.default_rng(0)
    dist = rng.dirichlet(np.ones(n_bins), size=4).astype(np.float32)
    # leave atoms in-range so no mass is lost to clipping
    shifted = np.broadcast_to(support, (4, n_bins)).astype(np.float32)
    out = categorical_projection(jnp.asarray(dist), jnp.asarray(shifted), vmin, vmax, delta, n_bins)
    np.testing.assert_allclose(np.asarray(out).sum(axis=1), 1.0, atol=1e-5)


def test_projection_point_mass_on_atom():
    support, vmin, vmax, delta, n_bins = make_support()
    # point mass on atom index 2, target value exactly on atom 7
    dist = np.zeros((1, n_bins), dtype=np.float32)
    dist[0, 2] = 1.0
    shifted = np.zeros((1, n_bins), dtype=np.float32)
    shifted[0, 2] = support[7]
    out = np.asarray(
        categorical_projection(jnp.asarray(dist), jnp.asarray(shifted), vmin, vmax, delta, n_bins)
    )
    assert np.isclose(out[0, 7], 1.0, atol=1e-5)
    assert np.isclose(out.sum(), 1.0, atol=1e-5)


def test_projection_midpoint_splits_50_50():
    support, vmin, vmax, delta, n_bins = make_support()
    dist = np.zeros((1, n_bins), dtype=np.float32)
    dist[0, 0] = 1.0
    midpoint = (support[3] + support[4]) / 2.0
    shifted = np.zeros((1, n_bins), dtype=np.float32)
    shifted[0, 0] = midpoint
    out = np.asarray(
        categorical_projection(jnp.asarray(dist), jnp.asarray(shifted), vmin, vmax, delta, n_bins)
    )
    assert np.isclose(out[0, 3], 0.5, atol=1e-5)
    assert np.isclose(out[0, 4], 0.5, atol=1e-5)


def test_projection_clips_at_support_edges():
    support, vmin, vmax, delta, n_bins = make_support()
    dist = np.zeros((1, n_bins), dtype=np.float32)
    dist[0, 5] = 1.0
    shifted = np.zeros((1, n_bins), dtype=np.float32)
    shifted[0, 5] = 1e6  # way above vmax -> clip to top atom
    out = np.asarray(
        categorical_projection(jnp.asarray(dist), jnp.asarray(shifted), vmin, vmax, delta, n_bins)
    )
    assert np.isclose(out[0, n_bins - 1], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# greedy targets, gamma=0
# ---------------------------------------------------------------------------
def test_greedy_gamma_zero_categorical_is_projection_of_reward():
    backend = make_cat_backend()
    support, vmin, vmax, delta, n_bins = make_support()
    B, A = 3, 4
    rng = np.random.default_rng(1)
    next_dists = rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32)
    reward = np.array([[2.0], [-1.0], [0.0]], dtype=np.float32)
    not_term = np.ones((B, 1), dtype=np.float32)
    actions = np.zeros((B, 1), dtype=np.int32)

    out = np.asarray(
        distributional_td_target(
            next_dists=jnp.asarray(next_dists),
            actions=jnp.asarray(actions),
            reward=jnp.asarray(reward),
            not_terminated=jnp.asarray(not_term),
            gamma=0.0,
            backend=backend,
        )
    )
    # gamma=0: target atoms collapse to reward; projection of a point at reward.
    for i in range(B):
        expected = np.asarray(
            categorical_projection(
                jnp.ones((1, n_bins)) / n_bins,
                jnp.full((1, n_bins), reward[i, 0]),
                vmin,
                vmax,
                delta,
                n_bins,
            )
        )[0]
        np.testing.assert_allclose(out[i], expected, atol=1e-5)


def test_greedy_gamma_zero_hlgauss_is_to_probs_of_reward():
    hl = HLGaussTransform.build(-5, 5, 11)
    backend = HLGaussBackend(hl)
    B, A, n_bins = 3, 4, 11
    rng = np.random.default_rng(2)
    next_dists = rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32)
    reward = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
    not_term = np.ones((B, 1), dtype=np.float32)
    actions = np.zeros((B, 1), dtype=np.int32)

    out = np.asarray(
        distributional_td_target(
            next_dists=jnp.asarray(next_dists),
            actions=jnp.asarray(actions),
            reward=jnp.asarray(reward),
            not_terminated=jnp.asarray(not_term),
            gamma=0.0,
            backend=backend,
        )
    )
    expected = np.asarray(hl.to_probs(jnp.asarray(reward)))
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_greedy_categorical_mean_matches_clipped_bellman():
    # C51 projection only conserves the mean when no shifted atom clips at an
    # edge. Confine next_dists mass to interior atoms so gamma*atom + reward
    # stays strictly inside [vmin, vmax].
    backend = make_cat_backend()
    support, vmin, vmax, delta, n_bins = make_support()
    B, A = 5, 4
    rng = np.random.default_rng(3)
    raw = rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32)
    interior = np.zeros_like(raw)
    interior[:, :, 3:8] = raw[:, :, 3:8]  # atoms in [-2, 2]
    next_dists = (interior / interior.sum(axis=2, keepdims=True)).astype(np.float32)
    reward = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
    not_term = np.ones((B, 1), dtype=np.float32)
    actions = np.zeros((B, 1), dtype=np.int32)
    gamma = 0.9

    out = distributional_td_target(
        next_dists=jnp.asarray(next_dists),
        actions=jnp.asarray(actions),
        reward=jnp.asarray(reward),
        not_terminated=jnp.asarray(not_term),
        gamma=gamma,
        backend=backend,
    )
    mean_target = np.asarray(jnp.sum(out * jnp.asarray(support), axis=1))

    q_next = (next_dists * support).sum(axis=2)  # [B, A]
    max_q = q_next.max(axis=1)
    expected = np.clip(reward[:, 0] + gamma * max_q, vmin, vmax)
    # bracket the mean: projection conserves mean while in-range (no edge clip here)
    np.testing.assert_allclose(mean_target, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# CHARACTERIZATION: canonical c51.py munchausen (categorical, distribution-space)
# ---------------------------------------------------------------------------
def _canonical_c51_munchausen(
    next_dists,
    online_next_dists,
    behavior_dists,
    actions,
    reward,
    not_term,
    gamma,
    support,
    vmin,
    vmax,
    delta,
    n_bins,
    alpha,
    tau,
    double_q,
):
    """Inline replica of c51.py `_target` munchausen branch."""
    bar = support

    def tdist(next_distribution, target_categorial):
        Tz = jnp.clip(target_categorial, vmin, vmax)
        B = ((Tz - vmin) / delta).astype(jnp.float32)
        L = jnp.floor(B).astype(jnp.int32)
        H = jnp.ceil(B).astype(jnp.int32)

        def project_one(p, b, _l, _u):
            exact = _l == _u
            m = jnp.zeros((n_bins,), dtype=p.dtype)
            w_l = jnp.where(exact, p, p * (_u.astype(jnp.float32) - b))
            w_u = jnp.where(exact, jnp.zeros_like(p), p * (b - _l.astype(jnp.float32)))
            m = m.at[_l].add(w_l)
            m = m.at[_u].add(w_u)
            return m

        return jax.vmap(project_one, in_axes=(0, 0, 0, 0))(next_distribution, B, L, H)

    if double_q:
        next_action_q = jnp.sum(online_next_dists * bar, axis=2)
    else:
        next_action_q = jnp.sum(next_dists * bar, axis=2)

    next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, tau)
    pi_next = jax.nn.softmax(next_sub_q / tau)
    next_categorials = bar - jnp.expand_dims(tau_log_pi_next, axis=2)

    q_k_targets = jnp.sum(behavior_dists * bar, axis=2)
    _, tau_log_pi = q_log_pi(q_k_targets, tau)
    addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=2), axis=1)
    rewards = reward + alpha * jnp.clip(addon, -1, 0)

    target_categorials = jnp.expand_dims(
        gamma * not_term, axis=2
    ) * next_categorials + jnp.expand_dims(rewards, axis=2)
    target_distributions = jax.vmap(tdist, in_axes=(1, 1), out_axes=1)(
        next_dists, target_categorials
    )
    return jnp.sum(jnp.expand_dims(pi_next, axis=2) * target_distributions, axis=1)


def test_characterization_categorical_munchausen_matches_canonical():
    backend = make_cat_backend()
    support_np, vmin, vmax, delta, n_bins = make_support()
    support = jnp.asarray(support_np)
    B, A = 4, 5
    rng = np.random.default_rng(10)
    next_dists = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    online_next = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    behavior = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    reward = jnp.asarray(rng.uniform(-1, 1, size=(B, 1)).astype(np.float32))
    not_term = jnp.asarray((rng.random((B, 1)) > 0.3).astype(np.float32))
    actions = jnp.asarray(rng.integers(0, A, size=(B, 1)).astype(np.int32))
    gamma, alpha, tau = 0.97, 0.9, 0.03

    spec = MunchausenSpec(alpha=alpha, tau=tau)

    for double_q in (False, True):
        out = distributional_td_target(
            next_dists=next_dists,
            online_next_dists=online_next if double_q else None,
            behavior_dists=behavior,
            actions=actions,
            reward=reward,
            not_terminated=not_term,
            gamma=gamma,
            munchausen=spec,
            backend=backend,
        )
        expected = _canonical_c51_munchausen(
            next_dists,
            online_next,
            behavior,
            actions[:, :, None],  # canonical squeezes axis=2
            reward,
            not_term,
            gamma,
            support,
            vmin,
            vmax,
            delta,
            n_bins,
            alpha,
            tau,
            double_q,
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# CHARACTERIZATION: canonical hl_gauss_c51.py munchausen (scalar-space)
# ---------------------------------------------------------------------------
def _canonical_hlgauss_munchausen(
    hl,
    next_dists,
    online_next_dists,
    behavior_dists,
    actions,
    reward,
    not_term,
    gamma,
    alpha,
    tau,
    double_q,
):
    """Inline replica of hl_gauss_c51.py `_target` munchausen branch."""
    next_q = hl.to_scalar(next_dists)
    if double_q:
        next_action_q = hl.to_scalar(online_next_dists)
        next_sub_q, tau_log_pi_next = q_log_pi(next_action_q, tau)
    else:
        next_sub_q, tau_log_pi_next = q_log_pi(next_q, tau)
    pi_next = jax.nn.softmax(next_sub_q / tau)
    next_vals = jnp.sum(pi_next * (next_q - tau_log_pi_next), axis=1, keepdims=True) * not_term

    q_k_targets = hl.to_scalar(behavior_dists)
    _, tau_log_pi = q_log_pi(q_k_targets, tau)
    addon = jnp.take_along_axis(tau_log_pi, jnp.squeeze(actions, axis=1), axis=1)
    rewards = reward + alpha * jnp.clip(addon, -1, 0)
    target_q = next_vals * gamma + rewards
    return hl.to_probs(target_q)


def test_characterization_hlgauss_munchausen_matches_canonical():
    hl = HLGaussTransform.build(-5, 5, 11)
    backend = HLGaussBackend(hl)
    B, A, n_bins = 4, 5, 11
    rng = np.random.default_rng(11)
    next_dists = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    online_next = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    behavior = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    reward = jnp.asarray(rng.uniform(-1, 1, size=(B, 1)).astype(np.float32))
    not_term = jnp.asarray((rng.random((B, 1)) > 0.3).astype(np.float32))
    actions = jnp.asarray(rng.integers(0, A, size=(B, 1)).astype(np.int32))
    gamma, alpha, tau = 0.97, 0.9, 0.03
    spec = MunchausenSpec(alpha=alpha, tau=tau)

    for double_q in (False, True):
        out = distributional_td_target(
            next_dists=next_dists,
            online_next_dists=online_next if double_q else None,
            behavior_dists=behavior,
            actions=actions,
            reward=reward,
            not_terminated=not_term,
            gamma=gamma,
            munchausen=spec,
            backend=backend,
        )
        expected = _canonical_hlgauss_munchausen(
            hl,
            next_dists,
            online_next,
            behavior,
            actions[:, :, None],  # canonical squeezes axis=1 of [B,1,1] -> [B,1]
            reward,
            not_term,
            gamma,
            alpha,
            tau,
            double_q,
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# gamma scalar vs per-sample array [B,1]
# ---------------------------------------------------------------------------
def test_gamma_array_matches_scalar_when_uniform_categorical():
    backend = make_cat_backend()
    _, _, _, _, n_bins = make_support()
    B, A = 3, 4
    rng = np.random.default_rng(20)
    next_dists = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    reward = jnp.asarray(rng.uniform(-1, 1, size=(B, 1)).astype(np.float32))
    not_term = jnp.ones((B, 1), dtype=jnp.float32)
    actions = jnp.zeros((B, 1), dtype=jnp.int32)

    out_scalar = distributional_td_target(
        next_dists=next_dists,
        actions=actions,
        reward=reward,
        not_terminated=not_term,
        gamma=0.9,
        backend=backend,
    )
    out_array = distributional_td_target(
        next_dists=next_dists,
        actions=actions,
        reward=reward,
        not_terminated=not_term,
        gamma=jnp.full((B, 1), 0.9),
        backend=backend,
    )
    np.testing.assert_allclose(np.asarray(out_scalar), np.asarray(out_array), atol=1e-6)


def test_gamma_array_per_sample_hlgauss():
    hl = HLGaussTransform.build(-5, 5, 11)
    backend = HLGaussBackend(hl)
    B, A, n_bins = 3, 4, 11
    rng = np.random.default_rng(21)
    next_dists = jnp.asarray(rng.dirichlet(np.ones(n_bins), size=(B, A)).astype(np.float32))
    reward = jnp.asarray(rng.uniform(-1, 1, size=(B, 1)).astype(np.float32))
    not_term = jnp.ones((B, 1), dtype=jnp.float32)
    actions = jnp.zeros((B, 1), dtype=jnp.int32)
    gammas = jnp.asarray(np.array([[0.9], [0.8], [0.7]], dtype=np.float32))

    out = distributional_td_target(
        next_dists=next_dists,
        actions=actions,
        reward=reward,
        not_terminated=not_term,
        gamma=gammas,
        backend=backend,
    )
    # mirror scalar greedy hlgauss with per-sample gamma inline
    next_q = np.asarray(hl.to_scalar(next_dists))
    greedy = next_q.argmax(axis=1)
    nv = next_q[np.arange(B), greedy][:, None]
    target_q = nv * np.asarray(gammas) + np.asarray(reward)
    expected = np.asarray(hl.to_probs(jnp.asarray(target_q)))
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# QuantileHuberLosses argument-order convention: (target, pred)
#
# QuantileHuberLosses is ASYMMETRIC: error = target_tile - q_tile and
# weight = |quantile - (error < 0)|, so swapping the first two args inverts the
# per-quantile gradient weighting. The codebase canon is (target, pred) — see
# qrdqn.py, iqn.py, apex_iqn.py `_loss`, tqc.py, fqf.py. This locks that order
# so a future swap (pred, target) is detected.
# ---------------------------------------------------------------------------
def test_quantile_huber_loss_arg_order_convention():
    delta = 1.0
    quantile = jnp.asarray([0.125, 0.375, 0.625, 0.875]).reshape(1, 1, 4)

    # (a) ASYMMETRY: target != pred -> (target, pred) != (pred, target).
    # docstring shapes: target_tile (B, num_tau_prime, 1), q_tile (B, 1, num_tau).
    target_a = jnp.asarray([2.0, -1.0, 0.5, 3.0]).reshape(1, 4, 1)
    pred_a = jnp.asarray([0.0, 1.0, -2.0, 0.5]).reshape(1, 1, 4)
    loss_canonical = QuantileHuberLosses(target_a, pred_a, quantile, delta)
    # swap operands; reshape so each lands in the slot the docstring expects.
    loss_swapped = QuantileHuberLosses(
        pred_a.reshape(1, 4, 1), target_a.reshape(1, 1, 4), quantile, delta
    )
    assert not np.allclose(
        np.asarray(loss_canonical), np.asarray(loss_swapped)
    ), "QuantileHuberLosses is asymmetric in (target, pred); swapping must change the loss"

    # (b) SEMANTICS: target > pred across all quantiles -> error > 0 -> error_neg = 0
    # -> weight = |quantile - 0| = quantile. So d(loss)/d(pred) scales with quantile:
    # high quantiles are weighted more strongly than low ones.
    target = jnp.asarray([3.0, 3.5, 4.0, 4.5]).reshape(1, 4, 1)
    pred = jnp.asarray([0.0, 0.5, 1.0, 1.5]).reshape(1, 1, 4)

    def loss_of_pred(p):
        return jnp.sum(QuantileHuberLosses(target, p, quantile, delta))

    grad = np.asarray(jax.grad(loss_of_pred)(pred)).reshape(-1)
    # all errors land in the huber linear region (|error| > delta), so the only
    # per-atom modulation is the quantile weight; gradient sign is negative
    # because error = target - pred decreases as pred grows.
    assert np.all(grad < 0.0), "raising pred toward a larger target must lower the loss"
    abs_grad = np.abs(grad)
    assert np.all(
        np.diff(abs_grad) > 0.0
    ), "canonical (target, pred) loss must weight high quantiles more than low ones"
    # not a tautology: the magnitudes equal the quantiles themselves here.
    np.testing.assert_allclose(abs_grad, np.asarray(quantile).reshape(-1), atol=1e-6)
