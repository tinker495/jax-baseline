"""Pure distributional TD-target operator and backends.

This module owns the *scalar scaffolding* shared by every distributional Q-Net
``_target`` (C51, apex_c51, HL-Gauss C51/SPR/BBF): double-Q action selection,
Munchausen reward shaping, and the action-mixture weights. The
representation-specific projection step is delegated to a **Distributional
backend** (``CategoricalBackend`` or ``HLGaussBackend``).

Everything here is pure (no network calls, no ``self``, no PRNG) so it is
unit-testable with synthetic arrays. Callers compute next-state / current-state
distributions with their own ``get_q`` and pass them in.

See ``CONTEXT.md`` for the locked domain vocabulary and canonical-semantics
decisions (the canonical Munchausen form is ``c51.py`` / ``hl_gauss_c51.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import jax
import jax.numpy as jnp

from jax_baselines.math.policy_math import q_log_pi


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


def categorical_projection(
    next_dist: jax.Array,
    shifted_atom_values: jax.Array,
    support_min: float,
    support_max: float,
    delta: float,
    n_bins: int,
) -> jax.Array:
    """Pure C51 projection of one batch of distributions onto the fixed support.

    Redistributes the probability mass of ``next_dist`` onto the ``n_bins`` fixed
    atoms after each atom has been shifted by the Bellman update to
    ``shifted_atom_values``. Mirrors ``c51.py``'s ``tdist`` / ``project_one``
    (the ``l == u`` integer-bracket form; numerically equivalent to the
    ``l -= 1 / u += 1`` form used in apex_c51).

    Args:
        next_dist: ``[B, n_bins]`` source probability mass per atom.
        shifted_atom_values: ``[B, n_bins]`` Bellman-shifted value of each atom.
        support_min: lower edge of the value support.
        support_max: upper edge of the value support.
        delta: spacing between adjacent atoms.
        n_bins: number of atoms.

    Returns:
        ``[B, n_bins]`` projected probability mass.
    """

    def project_row(p: jax.Array, target_values: jax.Array) -> jax.Array:
        tz = jnp.clip(target_values, support_min, support_max)
        b = ((tz - support_min) / delta).astype(jnp.float32)
        lower = jnp.floor(b).astype(jnp.int32)
        upper = jnp.ceil(b).astype(jnp.int32)

        def project_one(p_i, b_i, l_i, u_i):
            exact = l_i == u_i
            m = jnp.zeros((n_bins,), dtype=p_i.dtype)
            w_l = jnp.where(exact, p_i, p_i * (u_i.astype(jnp.float32) - b_i))
            w_u = jnp.where(exact, jnp.zeros_like(p_i), p_i * (b_i - l_i.astype(jnp.float32)))
            m = m.at[l_i].add(w_l)
            m = m.at[u_i].add(w_u)
            return m

        return jnp.sum(
            jax.vmap(project_one, in_axes=(0, 0, 0, 0))(p, b, lower, upper),
            axis=0,
        )

    return jax.vmap(project_row, in_axes=(0, 0))(next_dist, shifted_atom_values)


class DistributionalBackend(Protocol):
    """Seam between the shared target operator and a value representation."""

    def action_values(self, dist: jax.Array) -> jax.Array:
        """Map ``[B, A, bins]`` distributions to ``[B, A]`` scalar action values."""
        ...

    def project(
        self,
        next_dists: jax.Array,
        weights: jax.Array,
        entropy_shift: Optional[jax.Array],
        reward: jax.Array,
        not_terminated: jax.Array,
        gamma: jax.Array,
    ) -> jax.Array:
        """Build the ``[B, bins]`` target distribution from per-action next dists.

        Args:
            next_dists: ``[B, A, bins]`` next-state distributions.
            weights: ``[B, A]`` action-mixture weights (one-hot for greedy,
                ``pi_next`` for Munchausen).
            entropy_shift: ``[B, A]`` per-action support shift (``tau_log_pi_next``)
                or ``None`` for greedy (treated as 0).
            reward: ``[B, 1]`` (possibly Munchausen-shaped) reward.
            not_terminated: ``[B, 1]`` terminal mask.
            gamma: scalar or ``[B, 1]`` discount.

        Returns:
            ``[B, bins]`` target distribution.
        """
        ...


@dataclass(frozen=True, eq=False)
class CategoricalBackend:
    """Fixed-atom C51 backend. Mixes Munchausen targets in *distribution space*."""

    support: jax.Array
    support_min: float
    support_max: float
    delta: float
    n_bins: int

    def action_values(self, dist: jax.Array) -> jax.Array:
        # dist: [B, A, bins], support: [bins] -> [B, A]
        return jnp.sum(dist * self.support, axis=2)

    def project(
        self,
        next_dists: jax.Array,
        weights: jax.Array,
        entropy_shift: Optional[jax.Array],
        reward: jax.Array,
        not_terminated: jax.Array,
        gamma: jax.Array,
    ) -> jax.Array:
        if entropy_shift is None:
            entropy_shift = jnp.zeros(weights.shape, dtype=self.support.dtype)
        # per-action shifted support: [B, A, bins]
        shifted_support = self.support[None, None, :] - entropy_shift[:, :, None]
        # broadcast gamma/not_terminated/reward to [B, 1, 1]
        scale = jnp.expand_dims(gamma * not_terminated, axis=2)
        target_atoms = scale * shifted_support + jnp.expand_dims(reward, axis=2)

        def project_action(next_dist_a, target_atoms_a):
            return categorical_projection(
                next_dist_a,
                target_atoms_a,
                self.support_min,
                self.support_max,
                self.delta,
                self.n_bins,
            )

        # vmap over the action axis (axis 1)
        dist_per_action = jax.vmap(project_action, in_axes=(1, 1), out_axes=1)(
            next_dists, target_atoms
        )
        return jnp.sum(jnp.expand_dims(weights, axis=2) * dist_per_action, axis=1)


@dataclass(frozen=True, eq=False)
class HLGaussBackend:
    """HL-Gauss backend. Mixes Munchausen targets in *scalar space*."""

    hl_gauss: HLGaussTransform

    def action_values(self, dist: jax.Array) -> jax.Array:
        # dist: [B, A, bins] -> [B, A]
        return self.hl_gauss.to_scalar(dist)

    def project(
        self,
        next_dists: jax.Array,
        weights: jax.Array,
        entropy_shift: Optional[jax.Array],
        reward: jax.Array,
        not_terminated: jax.Array,
        gamma: jax.Array,
    ) -> jax.Array:
        next_q = self.hl_gauss.to_scalar(next_dists)  # [B, A]
        if entropy_shift is None:
            entropy_shift = jnp.zeros(next_q.shape, dtype=next_q.dtype)
        next_vals = (
            jnp.sum(weights * (next_q - entropy_shift), axis=1, keepdims=True) * not_terminated
        )
        target_q = next_vals * gamma + reward
        return self.hl_gauss.to_probs(target_q)


@dataclass(frozen=True)
class MunchausenSpec:
    """Munchausen reward-shaping hyperparameters."""

    alpha: float
    tau: float


def distributional_td_target(
    *,
    next_dists: jax.Array,
    actions: jax.Array,
    reward: jax.Array,
    not_terminated: jax.Array,
    gamma: jax.Array,
    backend: DistributionalBackend,
    online_next_dists: Optional[jax.Array] = None,
    behavior_dists: Optional[jax.Array] = None,
    munchausen: Optional[MunchausenSpec] = None,
) -> jax.Array:
    """Compute the distributional TD target distribution ``[B, bins]``.

    Owns the scalar scaffolding shared by every distributional ``_target``;
    delegates the representation-specific projection to ``backend``.

    Double-Q is expressed purely by which distributions the caller passes:
    ``online_next_dists`` (online next-state dists) drives action selection when
    present, and ``behavior_dists`` carries the current-state distributions used
    by the Munchausen addon (online if double-Q else target). When they are
    ``None`` the target-network ``next_dists`` is used instead.

    Args:
        next_dists: ``[B, A, bins]`` target-network next-state distributions.
        actions: ``[B, 1]`` taken actions (Munchausen addon index).
        reward: ``[B, 1]`` reward.
        not_terminated: ``[B, 1]`` terminal mask.
        gamma: scalar or ``[B, 1]`` discount (broadcasts either way).
        backend: distributional backend adapter.
        online_next_dists: ``[B, A, bins]`` online next-state dists for double-Q
            action selection; falls back to ``next_dists`` when ``None``.
        behavior_dists: ``[B, A, bins]`` current-state dists for the Munchausen
            addon; required when ``munchausen`` is set.
        munchausen: Munchausen spec, or ``None`` for the greedy target.

    Returns:
        ``[B, bins]`` target distribution.
    """
    selection_dists = next_dists if online_next_dists is None else online_next_dists

    if munchausen is None:
        q_next = backend.action_values(selection_dists)
        greedy = jnp.argmax(q_next, axis=1)
        weights = jax.nn.one_hot(greedy, q_next.shape[1], dtype=next_dists.dtype)
        return backend.project(
            next_dists,
            weights,
            None,
            reward,
            not_terminated,
            gamma,
        )

    tau = munchausen.tau
    q_next = backend.action_values(selection_dists)
    sub_q, tau_log_pi_next = q_log_pi(q_next, tau)
    pi_next = jax.nn.softmax(sub_q / tau)

    if behavior_dists is None:
        raise ValueError("munchausen target requires behavior_dists")
    q_k = backend.action_values(behavior_dists)
    _, tau_log_pi = q_log_pi(q_k, tau)
    addon = jnp.take_along_axis(tau_log_pi, actions, axis=1)
    shaped_reward = reward + munchausen.alpha * jnp.clip(addon, -1, 0)

    return backend.project(
        next_dists,
        pi_next,
        tau_log_pi_next,
        shaped_reward,
        not_terminated,
        gamma,
    )
