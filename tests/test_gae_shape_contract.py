"""Regression tests for the GAE shape contract (2026-06-16).

The on-policy ``EpochBuffer`` stores one scalar reward/terminated/truncated per
step, so after stacking they reach ``get_gaes`` as flat ``[T]`` vectors, while
the critic keeps a trailing unit dim and supplies ``[T, 1]`` values. Commit
4d7c390 migrated the buffer to this scalar-per-step form but left ``get_gaes``
assuming the old cpprb ``[T, 1]`` reward shape. The mismatched ``[T]`` reward
broadcast against ``[T, 1]`` values into a ``[T, T]`` ``deltas`` matrix, and the
backward scan blew up its carry shape (``float32[1]`` -> ``float32[T]``).

The pre-existing returns tests fed every input through a ``col()`` reshape to
``[T, 1]``, so they never exercised this flat-vs-column mismatch. These tests
pin the real production shapes: PPO/TPPO/SPO all call ``get_gaes`` this way.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.math.returns import get_gaes

GAMMA = 0.9
LAMBDA = 0.95


def _ref_gae(rewards, terminateds, truncateds, values, next_values):
    # Independent scalar reference for the backward GAE recursion.
    advs = [0.0] * len(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * (1.0 - terminateds[t]) * next_values[t] - values[t]
        last = delta + GAMMA * LAMBDA * (1.0 - terminateds[t]) * (1.0 - truncateds[t]) * last
        advs[t] = last
    return advs


def test_get_gaes_flat_rewards_against_column_values():
    # Production shapes: rewards/terminateds/truncateds flat [T], values [T, 1].
    rewards = [1.0, 2.0, 3.0, 4.0]
    terminateds = [0, 0, 1, 0]
    truncateds = [0, 1, 0, 0]
    values = [0.5, 0.25, 0.75, 0.5]
    next_values = [1.0, 1.0, 1.0, 1.0]

    adv = get_gaes(
        jnp.array(rewards, dtype=jnp.float32),
        jnp.array(terminateds, dtype=jnp.float32),
        jnp.array(truncateds, dtype=jnp.float32),
        jnp.array(values, dtype=jnp.float32).reshape(-1, 1),
        jnp.array(next_values, dtype=jnp.float32).reshape(-1, 1),
        GAMMA,
        LAMBDA,
    )

    # The bug produced (4, 4); the contract is one advantage per step, [T, 1].
    assert adv.shape == (4, 1)
    expected = _ref_gae(rewards, terminateds, truncateds, values, next_values)
    assert np.allclose(np.asarray(adv).reshape(-1), expected, atol=1e-5)


def test_get_gaes_vmapped_over_workers_matches_value_shape():
    # The actual call site: jax.vmap(get_gaes) over the worker axis, with the
    # exact shapes _preprocess builds -- rewards [W, T], values [W, T, 1].
    workers, steps = 8, 16
    rewards = jnp.ones((workers, steps), dtype=jnp.float32)
    terminateds = jnp.zeros((workers, steps), dtype=jnp.float32)
    truncateds = jnp.zeros((workers, steps), dtype=jnp.float32)
    values = jnp.ones((workers, steps, 1), dtype=jnp.float32)
    next_values = jnp.ones((workers, steps, 1), dtype=jnp.float32)

    adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
        rewards, terminateds, truncateds, values, next_values, GAMMA, LAMBDA
    )

    # Per-worker advantages stay [W, T, 1] so the downstream vstack lines up with
    # vstacked values for ``targets = value + adv``.
    assert adv.shape == (workers, steps, 1)
    assert jnp.vstack(adv).shape == jnp.vstack(values).shape
    assert bool(jnp.all(jnp.isfinite(adv)))
