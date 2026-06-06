"""TD7-style per-episode checkpoint schedule for the off-policy rollout families.

Sibling to :mod:`jax_baselines.common.rollout`: where :class:`RolloutEngine`
owns the environment-interaction loop, :class:`CheckpointController` owns the
checkpoint *schedule* that loop drives at every episode boundary. The two meet
only at the :class:`RolloutSpec` checkpoint seam — the engine calls
``spec.checkpoint_on_episode_end`` (bound to :meth:`CheckpointController.on_episode_end`)
and never sees the schedule's internals.

The schedule was previously copy-pasted into the Q-Net and DPG base classes and
had drifted (the DPG copy fell through its warmup branch into a second training
pulse, and disagreed on the return contract). It now lives here once, with the
Q-Net warmup semantics adopted as canonical: during warmup the controller
snapshots, fires a single pulse, and returns immediately.

Everything family-specific is injected:

- ``snapshot`` captures eval parameters/state (the only thing that varies
  across families behind the seam);
- ``log_metric`` records ``ckpt/*`` series (late-bound so it tolerates a logger
  that is created after the controller).

The controller owns all schedule runtime state and serializes it through
:meth:`to_state` / :meth:`from_state` (the DPG family persists it across
save/load). The training-cadence residual is *not* owned here — it belongs to
:class:`~jax_baselines.common.rollout.CheckpointTrainPulse`.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from jax_baselines.common.statistics import compute_ckpt_window_stat


@dataclass(frozen=True)
class CheckpointSetup:
    """Resolved checkpoint configuration and constructed controller.

    Returned by :func:`make_checkpoint_controller` so callers can assign the
    resolved config values as instance attributes in a single unpacking step,
    without repeating the resolution ladder.
    """

    controller: "CheckpointController"
    quantile: float
    use_return_standardization: bool
    baseline_mode: str
    baseline_q: float
    gate_mode: str
    gate_q: float


def make_checkpoint_controller(
    *,
    use_checkpointing: bool,
    steps_before_checkpointing: int,
    max_eps_before_checkpointing: int,
    initial_checkpoint_window: int,
    ckpt_baseline_mode: str,
    ckpt_baseline_q: Optional[float],
    ckpt_gate_mode: Optional[str],
    ckpt_gate_q: Optional[float],
    snapshot: Callable[[], None],
    log_metric: Callable[[str, float, int], None],
) -> CheckpointSetup:
    """Resolve checkpoint config and construct a :class:`CheckpointController`.

    Encapsulates the resolution ladder that was duplicated verbatim in the DQN
    and DDPG base-class constructors.  The caller is responsible for assigning
    the returned :class:`CheckpointSetup` fields as instance attributes and for
    initialising ``_ckpt_update_residual = 0`` separately (it is mutable
    per-run state, not config).

    Args:
        use_checkpointing: Enable the TD7-style checkpoint schedule.
        steps_before_checkpointing: Warm-up steps before checkpointing
            activates (already clamped to ``learning_starts * 2`` by the
            caller).
        max_eps_before_checkpointing: Episode budget per checkpoint window.
        initial_checkpoint_window: Initial window size before the first enable.
        ckpt_baseline_mode: Statistic used to build the rolling baseline.
        ckpt_baseline_q: Quantile for baseline computation; defaults to the
            canonical ``quantile`` (0.2) when ``None``.
        ckpt_gate_mode: Statistic used for gating; defaults to
            ``ckpt_baseline_mode`` when ``None``.
        ckpt_gate_q: Explicit gate quantile; resolved from ``ckpt_gate_mode``
            when ``None``.
        snapshot: Callable that captures the current policy parameters.
        log_metric: Callable ``(key, value, step)`` forwarded to the run
            logger.

    Returns:
        A frozen :class:`CheckpointSetup` with the constructed controller and
        all resolved config values.
    """
    quantile: float = 0.2
    use_return_standardization: bool = False

    resolved_baseline_q: float = ckpt_baseline_q if ckpt_baseline_q is not None else quantile
    resolved_gate_mode: str = ckpt_gate_mode if ckpt_gate_mode is not None else ckpt_baseline_mode

    if ckpt_gate_q is not None:
        resolved_gate_q: float = ckpt_gate_q
    elif resolved_gate_mode == "median":
        resolved_gate_q = 0.5
    elif resolved_gate_mode in ("quantile", "min", "mean"):
        resolved_gate_q = resolved_baseline_q
    else:
        resolved_gate_q = quantile

    controller = CheckpointController(
        use_checkpointing=use_checkpointing,
        steps_before_checkpointing=steps_before_checkpointing,
        max_eps_before_checkpointing=max_eps_before_checkpointing,
        initial_window=initial_checkpoint_window,
        baseline_q=resolved_baseline_q,
        baseline_mode=ckpt_baseline_mode,
        use_return_standardization=use_return_standardization,
        snapshot=snapshot,
        log_metric=log_metric,
    )

    return CheckpointSetup(
        controller=controller,
        quantile=quantile,
        use_return_standardization=use_return_standardization,
        baseline_mode=ckpt_baseline_mode,
        baseline_q=resolved_baseline_q,
        gate_mode=resolved_gate_mode,
        gate_q=resolved_gate_q,
    )


class CheckpointController:
    """Per-episode checkpoint schedule shared by the off-policy local families."""

    def __init__(
        self,
        *,
        use_checkpointing: bool,
        steps_before_checkpointing: int,
        max_eps_before_checkpointing: int,
        initial_window: int,
        baseline_q: float,
        baseline_mode: str,
        use_return_standardization: bool,
        snapshot: Callable[[], None],
        log_metric: Callable[[str, float, int], None],
    ):
        # Configuration
        self.use_checkpointing = use_checkpointing
        self._steps_before_checkpointing = int(steps_before_checkpointing)
        self._max_eps_before_checkpointing = int(max_eps_before_checkpointing)
        self._baseline_q = baseline_q
        self._baseline_mode = baseline_mode
        self._use_return_standardization = use_return_standardization
        self._snapshot = snapshot
        self._log_metric = log_metric

        # Schedule runtime state (owned here; serialized for the DPG family)
        self._enabled = False
        self._eps_since_update = 0
        self._timesteps_since_update = 0
        self._max_eps_before_update = int(initial_window)
        self._returns_window: list = []
        self._baseline = -1e8
        self._last_update_step: Optional[int] = None
        self._update_count = 0

    # -- read-only views the agent needs (eval gating, progress description) --

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_update_step(self) -> Optional[int]:
        return self._last_update_step

    # -- schedule --

    def _maybe_enable(self, steps):
        if (
            self.use_checkpointing
            and (not self._enabled)
            and steps > self._steps_before_checkpointing
        ):
            # Relax the threshold slightly when entering checkpointing mode
            self._max_eps_before_update = self._max_eps_before_checkpointing
            self._enabled = True

    def _reset_window(self):
        self._eps_since_update = 0
        self._timesteps_since_update = 0
        self._returns_window = []

    def _log_snapshot_update(self, steps):
        self._last_update_step = int(steps)
        self._update_count += int(self._enabled)
        self._log_metric("ckpt/ckpt_baseline", float(self._baseline), int(steps))
        self._log_metric("ckpt/update_count", float(self._update_count), int(steps))

    def on_episode_end(self, steps, episode_return, episode_len, train_and_reset_callback=None):
        """Advance the checkpoint schedule at an episode boundary.

        Returns True when the episode did not trigger a checkpoint failure, and
        False when the window fell below the baseline (the Q-Net family uses this
        to gate ``env.true_reset()``; the DPG family ignores it).
        """
        if not self.use_checkpointing:
            return True

        self._eps_since_update += 1
        self._timesteps_since_update += int(episode_len)
        self._returns_window.append(float(episode_return))

        self._maybe_enable(steps)

        window_stat = compute_ckpt_window_stat(
            self._returns_window,
            self._baseline_q,
            self._use_return_standardization,
            self._baseline_mode,
        )
        if window_stat is None:
            return True
        self._log_metric("ckpt/window_stat", float(window_stat), int(steps))

        # Warmup phase: snapshot a baseline, fire one pulse, return.
        if not self._enabled:
            self._snapshot()
            self._fire(train_and_reset_callback, steps)
            self._reset_window()
            return True

        if window_stat < self._baseline:
            # Below baseline: signal failure (Q-Net gates true_reset on this).
            self._fire(train_and_reset_callback, steps)
            self._reset_window()
            return False

        # Enabled phase: end-of-window refresh with a training pulse.
        if self._eps_since_update >= self._max_eps_before_update:
            self._snapshot()
            self._baseline = window_stat
            self._log_snapshot_update(steps)
            self._fire(train_and_reset_callback, steps)
            self._reset_window()

        return True

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
