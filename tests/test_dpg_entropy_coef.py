"""Unit tests for the shared entropy-coefficient setup on the DPG family base.

`Deteministic_Policy_Gradient_Family._setup_entropy_coef` and `_train_ent_coef`
were consolidated out of SAC/TQC/CrossQ. These tests pin the four documented
branches (auto, auto_<x>, fixed numeric, invalid) and the temperature gradient
step, calling the methods on a bare carrier object so no env or JAX model is
built.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family


class _EntCarrier:
    """Minimal object exposing only what the entropy methods touch."""

    def __init__(self, ent_coef, target_entropy=0.0, ent_coef_learning_rate=1e-4):
        self._ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.ent_coef_learning_rate = ent_coef_learning_rate


def _setup(ent_coef):
    carrier = _EntCarrier(ent_coef)
    Deteministic_Policy_Gradient_Family._setup_entropy_coef(carrier)
    return carrier


def test_auto_enables_tuning_with_default_init():
    carrier = _setup("auto")
    assert carrier.auto_entropy is True
    np.testing.assert_allclose(float(carrier.log_ent_coef), np.log(1e-1), rtol=1e-6)


def test_auto_suffix_seeds_initial_value():
    # The suffix must satisfy log(value) > 0, i.e. value > 1.0 (preserved
    # invariant from the original SAC/TQC/CrossQ setup).
    carrier = _setup("auto_2.0")
    assert carrier.auto_entropy is True
    np.testing.assert_allclose(float(carrier.log_ent_coef), np.log(2.0), rtol=1e-6)


def test_auto_suffix_rejects_value_at_or_below_one():
    with pytest.raises(AssertionError, match="greater than 0"):
        _setup("auto_0.5")


def test_fixed_numeric_disables_tuning():
    carrier = _setup("0.2")
    assert carrier.auto_entropy is False
    np.testing.assert_allclose(float(carrier.log_ent_coef), np.log(0.2), rtol=1e-6)


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="Invalid value for ent_coef"):
        _setup("not_a_number")


def test_train_ent_coef_gradient_step():
    carrier = _EntCarrier("auto", target_entropy=-3.0, ent_coef_learning_rate=0.1)
    log_coef = jnp.asarray(np.log(0.1))
    log_prob = jnp.asarray([[-1.0], [-2.0]])

    updated = Deteministic_Policy_Gradient_Family._train_ent_coef(carrier, log_coef, log_prob)

    # loss = mean(exp(log_coef) * (target_entropy - log_prob)); d/d log_coef of
    # exp(log_coef)*c is exp(log_coef)*c, so grad = mean(exp(log_coef)*(te - lp)).
    ent_coef = np.exp(float(log_coef))
    expected_grad = np.mean(ent_coef * (-3.0 - np.array([-1.0, -2.0])))
    expected = float(log_coef) - 0.1 * expected_grad
    np.testing.assert_allclose(float(updated), expected, rtol=1e-5)


def test_train_ent_coef_is_jittable():
    carrier = _EntCarrier("auto", target_entropy=-1.0)
    fn = jax.jit(
        lambda lc, lp: Deteministic_Policy_Gradient_Family._train_ent_coef(carrier, lc, lp)
    )
    out = fn(jnp.asarray(0.0), jnp.asarray([[-1.0]]))
    assert np.isfinite(float(out))
