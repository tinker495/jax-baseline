"""Contract tests for jax_baselines.math.distributional.HLGaussTransform.

The HL-Gauss scalar<->distribution transform was copy-pasted (byte-identical)
into the HL_GAUSS variants of C51, BBF and SPR, and had no direct test. These
lock the interface: support/sigma construction, normalization, scalar round-trip,
the re-entrancy fix (build() never compounds sigma), and jit-compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.math.distributional import HLGaussTransform


def test_build_sets_support_edges_and_bin_scaled_sigma():
    t = HLGaussTransform.build(-250, 250, 51)
    # n_bins + 1 edges spanning [min, max]
    assert t.support.shape == (52,)
    assert float(t.support[0]) == -250.0
    assert float(t.support[-1]) == 250.0
    bin_width = float(t.support[1] - t.support[0])
    assert np.isclose(float(t.sigma), 0.75 * bin_width)


def test_build_is_pure_and_does_not_compound_sigma():
    # The old per-class code mutated self.sigma *= bin_width in setup_model, so a
    # second setup_model would double-scale. build() is pure; repeat calls match.
    t1 = HLGaussTransform.build(-10, 10, 100)
    t2 = HLGaussTransform.build(-10, 10, 100)
    assert float(t1.sigma) == float(t2.sigma)


def test_to_probs_rows_normalize_to_one():
    t = HLGaussTransform.build(-10, 10, 100)
    target = jnp.array([[2.0], [-3.0], [0.0], [7.5]])
    probs = t.to_probs(target)
    assert probs.shape == (4, 100)
    assert jnp.all(probs >= 0.0)
    assert jnp.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_scalar_round_trip_recovers_target_within_support():
    t = HLGaussTransform.build(-10, 10, 100)
    target = jnp.array([[2.0], [-3.0], [0.0], [5.5]])
    probs = t.to_probs(target)  # [batch, n_bins]
    recovered = t.to_scalar(probs[:, None, :])  # [batch, 1]
    assert recovered.shape == (4, 1)
    assert jnp.allclose(recovered[:, 0], target[:, 0], atol=0.2)


def test_transforms_run_under_jit_via_closure():
    # Mirrors real usage: the transform is captured by closure inside a jitted
    # train step, support/sigma bake in as constants.
    t = HLGaussTransform.build(-10, 10, 100)

    @jax.jit
    def round_trip(x):
        return t.to_scalar(t.to_probs(x)[:, None, :])

    out = round_trip(jnp.array([[1.0], [-4.0]]))
    assert jnp.allclose(out[:, 0], jnp.array([1.0, -4.0]), atol=0.2)
