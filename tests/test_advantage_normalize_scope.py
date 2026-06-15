"""Regression tests for the PPO-family advantage-normalization scope option.

The synchronous PG family (PPO / TPPO / SPO) keeps the ``gae_normalize`` bool
toggle (default ``False``, no normalization) and adds ``gae_normalize_scope``:

- ``"batch"`` (default, backward-compatible): standardize the whole flattened
  rollout once in ``_preprocess``.
- ``"minibatch"`` (PPO2/CleanRL-style): standardize each minibatch on its own
  stats inside the training loop, recomputed per minibatch per epoch.

Import-only: no Ray / env construction required.
"""

import argparse
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.cli import pg
from jax_baselines.common.returns import (
    ADVANTAGE_NORMALIZE_SCOPES,
    normalize_advantage,
    validate_advantage_normalize_scope,
)
from jax_baselines.PPO.ppo import PPO
from jax_baselines.SPO.spo import SPO
from jax_baselines.TPPO.tppo import TPPO


def test_normalize_advantage_standardizes_to_zero_mean_unit_std():
    out = np.asarray(normalize_advantage(jnp.array([[1.0], [2.0], [3.0], [4.0]]))).reshape(-1)
    assert abs(float(out.mean())) < 1e-5
    # eps in the denominator pulls std just under 1.
    assert abs(float(out.std()) - 1.0) < 1e-3


def test_batch_and_minibatch_scope_produce_different_normalization():
    # Two minibatches with different location/scale. Whole-batch uses global
    # mean/std; per-minibatch uses each minibatch's own stats, so the same
    # element is scaled differently under the two scopes.
    mb0 = jnp.array([[0.0], [2.0]])  # mean 1, std 1
    mb1 = jnp.array([[10.0], [14.0]])  # mean 12, std 2
    whole = np.asarray(normalize_advantage(jnp.vstack([mb0, mb1])))
    per_mb = np.asarray(jnp.vstack([normalize_advantage(mb0), normalize_advantage(mb1)]))
    assert not np.allclose(whole, per_mb)
    # Each minibatch is independently zero-mean under minibatch scope.
    assert np.allclose(per_mb.reshape(2, 2).mean(axis=1), 0.0, atol=1e-5)


def test_validate_scope_accepts_known_and_rejects_unknown():
    for scope in ADVANTAGE_NORMALIZE_SCOPES:
        assert validate_advantage_normalize_scope(scope) == scope
    with pytest.raises(ValueError):
        validate_advantage_normalize_scope("whole")


@pytest.mark.parametrize("cls", [PPO, TPPO, SPO])
def test_scope_guards_wire_to_correct_normalization_site(cls):
    # batch scope normalizes once in _preprocess; minibatch scope normalizes
    # inside the per-minibatch scan body of _train_step. The helper-level tests
    # can't see a dropped or batch<->minibatch-swapped guard, so pin the wiring
    # at the source level: each site must carry its own scope condition only.
    preprocess_src = inspect.getsource(cls._preprocess)
    train_src = inspect.getsource(cls._train_step)
    assert 'self.gae_normalize_scope == "batch"' in preprocess_src
    assert "normalize_advantage(adv)" in preprocess_src
    assert 'self.gae_normalize_scope == "minibatch"' in train_src
    assert "normalize_advantage(adv)" in train_src
    # guard against a copy-paste swap between the two sites.
    assert 'self.gae_normalize_scope == "minibatch"' not in preprocess_src
    assert 'self.gae_normalize_scope == "batch"' not in train_src


@pytest.mark.parametrize("cls", [PPO, TPPO, SPO])
def test_ppo_family_exposes_scope_with_backward_compatible_defaults(cls):
    params = inspect.signature(cls.__init__).parameters
    assert params["gae_normalize_scope"].default == "batch"
    # The on/off toggle stays a bool defaulting False so existing runs are unchanged.
    assert params["gae_normalize"].default is False


def test_cli_registers_and_forwards_scope():
    parser = argparse.ArgumentParser()
    pg.add_args(parser)

    default_ns = parser.parse_args([])
    assert default_ns.gae_normalize_scope == "batch"
    assert pg._ppo_like(default_ns)["gae_normalize_scope"] == "batch"

    mb_ns = parser.parse_args(["--gae_normalize_scope", "minibatch"])
    assert pg._ppo_like(mb_ns)["gae_normalize_scope"] == "minibatch"

    with pytest.raises(SystemExit):
        parser.parse_args(["--gae_normalize_scope", "bogus"])
