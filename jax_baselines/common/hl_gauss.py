from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, eq=False)
class HLGaussTransform:
    """HL-Gauss transform between scalar values and a categorical distribution.

    Maps a scalar onto a fixed support by integrating a Gaussian over each bin
    (``to_probs``), and recovers a scalar as the support-weighted mean of a
    distribution (``to_scalar``). This is the histogram-loss / Gaussian-smoothing
    representation shared by the HL-Gauss variants of C51, BBF and SPR.

    Build once via :meth:`build` and access through a held instance; ``support``
    and ``sigma`` are concrete arrays that bake in as constants under ``jax.jit``.

    support: shape ``[n_bins + 1]`` — the bin edges.
    sigma:   smoothing width, already scaled by the bin width.
    """

    support: jax.Array
    sigma: jax.Array

    @classmethod
    def build(cls, categorial_min, categorial_max, categorial_bar_n, sigma_ratio=0.75):
        support = jnp.linspace(
            float(categorial_min),
            float(categorial_max),
            categorial_bar_n + 1,
            dtype=jnp.float32,
        )
        bin_width = support[1] - support[0]
        return cls(support=support, sigma=sigma_ratio * bin_width)

    def to_probs(self, target: jax.Array) -> jax.Array:
        # target: [batch, 1] -> probs: [batch, n_bins]
        def f(target):
            cdf_evals = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        return jax.vmap(f)(target)

    def to_scalar(self, probs: jax.Array) -> jax.Array:
        # probs: [batch, n, n_bins] -> scalar: [batch, n]
        def f(probs):
            centers = (self.support[:-1] + self.support[1:]) / 2
            return jnp.sum(probs * centers)

        return jax.vmap(jax.vmap(f))(probs)
