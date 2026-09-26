"""TD7-style per-episode checkpoint schedule for the off-policy rollout families.

Sibling to :mod:`jax_baselines.core.rollout`: where :class:`RolloutEngine`
owns the environment-interaction loop, :class:`CheckpointController` owns the
checkpoint *schedule* that loop drives at every episode boundary. The two meet
only at the :class:`RolloutSpec` checkpoint seam — the engine calls
``spec.checkpoint_on_episode_end`` (bound to :meth:`CheckpointController.on_episode_end`)
and never sees the schedule's internals.

The schedule lives here once for the Q-Net and DPG base classes and follows TD7
(Fujimoto et al. 2023): short windows with a tracked baseline from the start, long
windows plus a one-time baseline relaxation later in the run.

Everything family-specific is injected:

- ``snapshot`` captures eval parameters/state (the only thing that varies
  across families behind the seam);
- ``log_metric`` is supplied to :meth:`on_episode_end` from the active run
  context and records ``ckpt/*`` series without retaining per-run state.

The controller owns all schedule runtime state and serializes it through
:meth:`to_state` / :meth:`from_state` (the DPG family persists it across
save/load). The training-cadence residual is *not* owned here — it belongs to
:class:`~jax_baselines.core.rollout.CheckpointTrainPulse`.
"""

from collections.abc import Callable
from copy import deepcopy

import jax
import numpy as np

_JAX_ARRAY_TYPE = getattr(jax, "Array", ())


def snapshot_pytree(tree):
    """Snapshot a parameter PyTree without copying immutable JAX array leaves."""
    return jax.tree_util.tree_map(_snapshot_leaf, tree)


def _snapshot_leaf(leaf):
    if _JAX_ARRAY_TYPE and isinstance(leaf, _JAX_ARRAY_TYPE):
        return leaf

    copy_leaf = getattr(leaf, "copy", None)
    if callable(copy_leaf):
        return copy_leaf()

    return deepcopy(leaf)


def make_checkpoint_scaffold(
    *,
    use_checkpointing: bool,
    checkpoint_start_fraction: float,
    max_eps_before_checkpointing: int,
    initial_checkpoint_window: int,
    ckpt_baseline_mode: str,
    ckpt_baseline_q: float | None,
    snapshot: Callable[[], None],
) -> "CheckpointController":
    """Resolve checkpoint config and build its controller.

    ``reset_weight`` (0.9, TD7's value) and the default quantile (0.2) are constants local to
    this factory.

    Args:
        use_checkpointing: Enable the TD7-style checkpoint schedule.
        checkpoint_start_fraction: Fraction of the run after which long windows begin (TD7
            starts them at 750k of 5M steps, 15%). Resolved to steps by
            :meth:`CheckpointController.schedule`.
        max_eps_before_checkpointing: Episodes per window once long windows begin.
        initial_checkpoint_window: Episodes per window before that.
        ckpt_baseline_mode: ``"min"`` (TD7: every episode must reach the baseline) or
            ``"quantile"`` (up to ``floor(q * window)`` episodes may fall below it).
        ckpt_baseline_q: Quantile ``q``; defaults to 0.2 when ``None``.
        snapshot: Callable that captures the current policy parameters.

    Returns:
        The :class:`CheckpointController` the base holds as ``self.ckpt``.
    """
    if ckpt_baseline_mode not in ("min", "quantile"):
        raise ValueError(
            f"ckpt_baseline_mode must be 'min' or 'quantile', got {ckpt_baseline_mode!r}"
        )
    if not 0.0 <= checkpoint_start_fraction <= 1.0:
        raise ValueError("checkpoint_start_fraction must be in [0, 1]")
    return CheckpointController(
        use_checkpointing=use_checkpointing,
        start_fraction=checkpoint_start_fraction,
        max_eps_before_checkpointing=max_eps_before_checkpointing,
        initial_window=initial_checkpoint_window,
        baseline_q=0.2 if ckpt_baseline_q is None else ckpt_baseline_q,
        baseline_mode=ckpt_baseline_mode,
        reset_weight=0.9,
        snapshot=snapshot,
    )


class CheckpointController:
    """Per-episode checkpoint schedule shared by the off-policy local families (TD7).

    Episodes form windows. A window fails, and trains without a checkpoint, once more than
    ``floor(q * window)`` of its episodes fall below the baseline (``q = 0`` for ``"min"``);
    a full window that holds checkpoints the policy, raises the baseline to its
    ``floor(q * window) + 1``-th lowest return, and trains. Windows start at
    ``initial_window`` episodes and grow to ``max_eps_before_checkpointing`` once
    ``start_fraction`` of the run has passed, when the baseline is relaxed by ``reset_weight``.
    """

    def __init__(
        self,
        *,
        use_checkpointing: bool,
        start_fraction: float,
        max_eps_before_checkpointing: int,
        initial_window: int,
        baseline_q: float,
        baseline_mode: str,
        reset_weight: float,
        snapshot: Callable[[], None],
    ):
        # Configuration
        self.use_checkpointing = use_checkpointing
        self.start_fraction = float(start_fraction)
        self.steps_before_checkpointing: int | None = None
        self.max_eps_before_checkpointing = int(max_eps_before_checkpointing)
        self.baseline_q = baseline_q
        self.baseline_mode = baseline_mode
        self.reset_weight = float(reset_weight)
        self._snapshot = snapshot

        # Schedule runtime state (owned here; serialized for the DPG family)
        self._enabled = False
        self._eps_since_update = 0
        self._timesteps_since_update = 0
        self._max_eps_before_update = int(initial_window)
        self._returns_window: list = []
        self._baseline = -1e8
        self._last_update_step: int | None = None
        self._update_count = 0

    # -- read-only views the agent needs (eval gating, progress description) --

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_update_step(self) -> int | None:
        return self._last_update_step

    # -- schedule --

    def schedule(self, total_timesteps):
        """Resolve the long-window start for a run of ``total_timesteps`` env steps."""
        self.steps_before_checkpointing = int(self.start_fraction * total_timesteps)

    def _maybe_enable(self, steps):
        if self.steps_before_checkpointing is None:
            raise RuntimeError("CheckpointController.schedule() must run before episodes end")
        if self.use_checkpointing and not self._enabled and steps > self.steps_before_checkpointing:
            # TD7 relaxes the baseline once as long windows begin.
            self._baseline *= self.reset_weight
            self._max_eps_before_update = self.max_eps_before_checkpointing
            self._enabled = True

    def _allowed_below(self):
        if self.baseline_mode == "min":
            return 0
        return int(self.baseline_q * self._max_eps_before_update)

    def _reset_window(self):
        self._eps_since_update = 0
        self._timesteps_since_update = 0
        self._returns_window = []

    def _log_snapshot_update(self, steps, log_metric):
        self._last_update_step = int(steps)
        self._update_count += int(self._enabled)
        if log_metric is not None:
            log_metric("ckpt/ckpt_baseline", float(self._baseline), int(steps))
            log_metric("ckpt/update_count", float(self._update_count), int(steps))

    def on_episode_end(
        self,
        steps,
        episode_return,
        episode_len,
        train_and_reset_callback=None,
        advance_criterion=True,
        log_metric=None,
    ):
        """Advance the checkpoint schedule at an episode boundary.

        Returns True when the episode did not fail the window, and False when it did (the
        rollout spec may use this to invoke an adapter-supplied forced reset).

        ``advance_criterion=False`` records the episode's timesteps toward the
        training pulse volume but leaves the assessment criterion untouched. The
        vectorized loop uses this so only the monitored worker's clean
        single-policy episode stream drives the checkpoint baseline, while every
        worker's collected timesteps still scale the pulse.
        """
        if not self.use_checkpointing:
            return True

        self._timesteps_since_update += int(episode_len)
        if not advance_criterion:
            return True

        self._eps_since_update += 1
        self._returns_window.append(float(episode_return))

        allowed = self._allowed_below()
        ordered = sorted(self._returns_window)
        if log_metric is not None:
            log_metric("ckpt/window_stat", ordered[min(allowed, len(ordered) - 1)], int(steps))

        if sum(value < self._baseline for value in ordered) > allowed:
            self._train_and_reset(train_and_reset_callback, steps)
            return False

        if self._eps_since_update >= self._max_eps_before_update:
            self._snapshot()
            self._baseline = ordered[allowed]
            self._log_snapshot_update(steps, log_metric)
            self._train_and_reset(train_and_reset_callback, steps)

        return True

    def _train_and_reset(self, callback, steps):
        self._fire(callback, steps)
        self._reset_window()
        # As in TD7, the pulse that crosses the start switches the windows that follow it.
        self._maybe_enable(steps)

    def _fire(self, callback, steps):
        if callable(callback):
            callback(steps, self._timesteps_since_update)

    # -- serialization (DPG family persists the schedule across save/load) --

    def to_state(self) -> dict:
        return {
            "checkpointing_enabled": np.asarray(self._enabled, dtype=np.bool_),
            "_ckpt_eps_since_update": np.asarray(self._eps_since_update, dtype=np.int32),
            "_ckpt_timesteps_since_update": np.asarray(
                self._timesteps_since_update, dtype=np.int64
            ),
            "_ckpt_baseline": np.asarray(self._baseline, dtype=np.float32),
            "_ckpt_update_count": np.asarray(self._update_count, dtype=np.int32),
            "_ckpt_max_eps_before_update": np.asarray(self._max_eps_before_update, dtype=np.int32),
            "_last_ckpt_update_step": (
                np.asarray(self._last_update_step, dtype=np.int64)
                if self._last_update_step is not None
                else np.asarray(-1, dtype=np.int64)
            ),
            "_ckpt_returns_window": np.asarray(self._returns_window, dtype=np.float32),
        }

    def from_state(self, state: dict):
        if "checkpointing_enabled" in state:
            self._enabled = bool(np.asarray(state["checkpointing_enabled"]).item())
        if "_ckpt_eps_since_update" in state:
            self._eps_since_update = int(np.asarray(state["_ckpt_eps_since_update"]).item())
        if "_ckpt_timesteps_since_update" in state:
            self._timesteps_since_update = int(
                np.asarray(state["_ckpt_timesteps_since_update"]).item()
            )
        if "_ckpt_baseline" in state:
            self._baseline = float(np.asarray(state["_ckpt_baseline"]).item())
        if "_ckpt_update_count" in state:
            self._update_count = int(np.asarray(state["_ckpt_update_count"]).item())
        if "_ckpt_max_eps_before_update" in state:
            self._max_eps_before_update = int(
                np.asarray(state["_ckpt_max_eps_before_update"]).item()
            )
        if "_last_ckpt_update_step" in state:
            last_update = int(np.asarray(state["_last_ckpt_update_step"]).item())
            self._last_update_step = None if last_update < 0 else last_update
        if "_ckpt_returns_window" in state:
            self._returns_window = np.asarray(state["_ckpt_returns_window"]).tolist()
