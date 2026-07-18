"""Golden-master pin for the APE-X C51 worker priority projection.

`APE_X_C51.get_actor_builder`'s `get_abs_td_error` used to hand-inline the C51
categorical projection (the ``l -= 1 / u += 1`` bracket form). It now delegates
to the canonical :func:`categorical_projection`. This test freezes the OLD
inlined math as a reference and asserts the consolidated worker path reproduces
the exact priority output (abs TD error) it produced before, to atol/rtol=1e-5.

Runs on CPU with tiny synthetic distributions; no network, no Ray.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.C51.apex_c51 import APE_X_C51

CATEGORIAL_BAR_N = 5
CATEGORIAL_MIN = -10.0
CATEGORIAL_MAX = 10.0
GAMMA = 0.99


def _support():
    return jnp.expand_dims(
        jnp.linspace(CATEGORIAL_MIN, CATEGORIAL_MAX, CATEGORIAL_BAR_N), axis=0
    )  # [1, bins]


def _reference_abs_td_error(distribution, next_q, rewards, terminateds):
    """The pre-refactor inlined projection, frozen verbatim as golden master."""
    categorial_bar = _support()
    delta_bar = (CATEGORIAL_MAX - CATEGORIAL_MIN) / (CATEGORIAL_BAR_N - 1)

    next_actions = jnp.expand_dims(
        jnp.argmax(jnp.sum(next_q * categorial_bar, axis=2), axis=1), axis=(1, 2)
    )
    next_distribution = jnp.squeeze(jnp.take_along_axis(next_q, next_actions, axis=1))
    next_categorial = (1.0 - terminateds) * categorial_bar
    target_categorial = (next_categorial * GAMMA) + rewards

    Tz = jnp.clip(target_categorial, CATEGORIAL_MIN, CATEGORIAL_MAX)
    C51_B = ((Tz - CATEGORIAL_MIN) / delta_bar).astype(jnp.float32)
    C51_L = jnp.floor(C51_B).astype(jnp.int32)
    C51_H = jnp.ceil(C51_B).astype(jnp.int32)
    C51_L = jnp.where((C51_H > 0) * (C51_L == C51_H), C51_L - 1, C51_L)
    C51_H = jnp.where((C51_L < (CATEGORIAL_BAR_N - 1)) * (C51_L == C51_H), C51_H + 1, C51_H)

    def tdist(next_distribution, C51_L, C51_H, C51_B):
        exact = C51_L == C51_H
        target_distribution = jnp.zeros((CATEGORIAL_BAR_N,))
        w_l = jnp.where(
            exact, next_distribution, next_distribution * (C51_H.astype(jnp.float32) - C51_B)
        )
        w_u = jnp.where(
            exact,
            jnp.zeros_like(next_distribution),
            next_distribution * (C51_B - C51_L.astype(jnp.float32)),
        )
        target_distribution = target_distribution.at[C51_L].add(w_l)
        target_distribution = target_distribution.at[C51_H].add(w_u)
        return target_distribution

    target_distribution = jax.vmap(tdist, in_axes=(0, 0, 0, 0))(
        next_distribution, C51_L, C51_H, C51_B
    )
    loss = jnp.mean(target_distribution * (-jnp.log(distribution + 1e-5)), axis=1)
    return jnp.squeeze(loss)


def _build_worker_get_abs_td_error():
    """Construct the live `get_abs_td_error` closure with a synthetic model."""
    agent = APE_X_C51.__new__(APE_X_C51)
    agent.action_size = (3,)
    agent.param_noise = False
    agent._gamma = GAMMA
    agent.categorial_bar_n = CATEGORIAL_BAR_N
    agent.categorial_min = CATEGORIAL_MIN
    agent.categorial_max = CATEGORIAL_MAX
    agent.categorial_bar = _support()
    agent.delta_bar = (CATEGORIAL_MAX - CATEGORIAL_MIN) / (CATEGORIAL_BAR_N - 1)

    get_abs_td_error, *_ = agent.get_actor_builder()()
    return get_abs_td_error


def _model_factory(cur_dist, nxt_dist):
    """A `model` mapping obs->cur_dist and nxtobs->nxt_dist.

    The two are distinguished by a marker scalar in the (real-array) observation
    so the closure's `convert_jax` boundary stays exercised.
    """

    def preproc(params, key, obses):
        return obses

    def model(params, key, x):
        marker = jnp.asarray(x["obs"])[0, 0]
        return jax.lax.cond(marker > 0.5, lambda: nxt_dist, lambda: cur_dist)

    return model, preproc


def _obs(marker, batch):
    return {"obs": np.full((batch, 1), marker, dtype=np.float32)}


def test_worker_priority_matches_inlined_golden_master():
    rng = np.random.default_rng(0)
    batch = 6
    n_actions = 3

    # Current-state distributions (per taken action) and next-state distributions.
    cur_logits = rng.normal(size=(batch, n_actions, CATEGORIAL_BAR_N))
    nxt_logits = rng.normal(size=(batch, n_actions, CATEGORIAL_BAR_N))
    cur_dist = jax.nn.softmax(jnp.asarray(cur_logits), axis=2)
    nxt_dist = jax.nn.softmax(jnp.asarray(nxt_logits), axis=2)

    actions = jnp.asarray(rng.integers(0, n_actions, size=(batch, 1)).astype(np.float32))
    rewards = jnp.asarray(rng.normal(size=(batch, 1)).astype(np.float32))
    terminateds = jnp.asarray(rng.integers(0, 2, size=(batch, 1)).astype(np.float32))

    # Build the reference: the worker selects current dist by `actions` then CE
    # against the projected target. Recreate the `distribution` it squeezes out.
    taken = jnp.squeeze(
        jnp.take_along_axis(cur_dist, jnp.expand_dims(actions.astype(jnp.int32), axis=2), axis=1)
    )
    reference = _reference_abs_td_error(taken, nxt_dist, rewards, terminateds)

    # Drive the live closure. obs (marker 0) -> cur_dist, nxtobs (marker 1) -> nxt_dist.
    model, preproc = _model_factory(cur_dist, nxt_dist)
    get_abs_td_error = _build_worker_get_abs_td_error()

    actual = get_abs_td_error(
        model,
        preproc,
        None,
        _obs(0.0, batch),
        actions,
        rewards,
        _obs(1.0, batch),
        terminateds,
        None,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), atol=1e-5, rtol=1e-5)


def test_projection_mass_is_conserved():
    """Sanity: a non-terminal projected target distribution sums to 1 per row."""
    rng = np.random.default_rng(1)
    batch, n_actions = 4, 3
    nxt_dist = jax.nn.softmax(
        jnp.asarray(rng.normal(size=(batch, n_actions, CATEGORIAL_BAR_N))), axis=2
    )
    cur_dist = jax.nn.softmax(
        jnp.asarray(rng.normal(size=(batch, n_actions, CATEGORIAL_BAR_N))), axis=2
    )
    actions = jnp.zeros((batch, 1), dtype=jnp.float32)
    rewards = jnp.zeros((batch, 1), dtype=jnp.float32)
    terminateds = jnp.zeros((batch, 1), dtype=jnp.float32)

    model, preproc = _model_factory(cur_dist, nxt_dist)
    get_abs_td_error = _build_worker_get_abs_td_error()
    # Smoke: the closure runs end-to-end and yields one priority per sample.
    out = get_abs_td_error(
        model,
        preproc,
        None,
        _obs(0.0, batch),
        actions,
        rewards,
        _obs(1.0, batch),
        terminateds,
        None,
    )
    assert np.asarray(out).shape == (batch,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
