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
    carrier = _setup("auto_0.01")
    assert carrier.auto_entropy is True
    np.testing.assert_allclose(float(carrier.log_ent_coef), np.log(0.01), rtol=1e-6)


@pytest.mark.parametrize("value", ["auto_0", "auto_-0.1"])
def test_auto_suffix_rejects_non_positive_value(value):
    with pytest.raises(AssertionError, match="greater than 0"):
        _setup(value)


def test_fixed_numeric_disables_tuning():
    carrier = _setup("0.2")
    assert carrier.auto_entropy is False
    np.testing.assert_allclose(float(carrier.log_ent_coef), np.log(0.2), rtol=1e-6)


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="Invalid value for ent_coef"):
        _setup("not_a_number")


def test_train_ent_coef_uses_adam_state_and_entropy_error():
    carrier = _EntCarrier("auto", target_entropy=1.0, ent_coef_learning_rate=0.1)
    Deteministic_Policy_Gradient_Family._setup_entropy_coef(carrier)
    log_coef = jnp.asarray(np.log(0.1))
    log_prob = jnp.asarray([[-2.0]])

    updated, opt_state = Deteministic_Policy_Gradient_Family._train_ent_coef(
        carrier, log_coef, carrier.opt_ent_coef_state, log_prob
    )

    assert float(updated) < float(log_coef)
    assert int(opt_state[0].count) == 1


def test_train_ent_coef_is_jittable():
    carrier = _EntCarrier("auto", target_entropy=-1.0)
    Deteministic_Policy_Gradient_Family._setup_entropy_coef(carrier)
    fn = jax.jit(
        lambda lc, state, lp: Deteministic_Policy_Gradient_Family._train_ent_coef(
            carrier, lc, state, lp
        )
    )
    out, _ = fn(jnp.asarray(0.0), carrier.opt_ent_coef_state, jnp.asarray([[-1.0]]))
    assert np.isfinite(float(out))
