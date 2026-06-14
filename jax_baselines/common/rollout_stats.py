"""Rollout episode statistics for the local off-policy / on-policy families.

The :class:`EpisodeTracker` owns the ``rollout/`` measurement namespace for the
single-process training families. Where ``eval`` produces one aggregated point
per ``eval_freq``, rollout episodes finish at irregular and (vectorized)
parallel times, so completed episodes are pushed into a fixed window and the
window mean is logged periodically.

The tracker is the only ``rollout/`` writer for these families; it reuses the
shared :func:`jax_baselines.common.eval.log_measurement` tag-writer so the
``rollout/`` leaves can never drift from the ``eval/`` leaves. The distributed
families keep their own server-side aggregation and do not use this tracker
(documented inconsistency, see ADR 0003).
"""

from collections import deque

import numpy as np

from jax_baselines.common.eval import log_measurement


class EpisodeTracker:
    """Windowed mean of behavior-policy training episodes, logged under ``rollout/``.

    Completed episodes are pushed via :meth:`record`; the window mean is logged
    at most once per ``log_interval`` env steps (throttled on the episode-end
    boundary, so an empty window is never logged). ``K=10`` matches the loss
    ``deque`` convention, trading smoothness for responsiveness.

    The engine that drives the rollout stays logger-free: it only calls
    :meth:`record`. The ``log_metric`` callable is injected by the agent and
    resolves the live ``logger_run`` lazily, so the tracker can be constructed
    once at agent ``__init__`` and is inert until a run binds the logger.
    """

    def __init__(self, log_metric, log_interval, window=10):
        self._log_metric = log_metric
        self._log_interval = log_interval
        self._reward = deque(maxlen=window)
        self._length = deque(maxlen=window)
        self._timeout = deque(maxlen=window)
        self._original = deque(maxlen=window)
        self._last_log_step = 0

    def record(self, steps, *, episode_reward, episode_length, timeout, original_reward=None):
        """Push one completed episode and flush the window if due.

        ``original_reward`` is recorded only when present (Atari unclipped
        score); ``timeout`` is the per-episode truncation flag (0/1) whose
        window mean is the truncation rate. Presence is assumed all-or-nothing
        per run (``ClipRewardEnv`` injects it on every Atari step and never
        otherwise), so the reward and original-reward windows stay aligned; an
        intermittently-present ``original_reward`` would let the two windows
        cover different episode spans.
        """
        self._reward.append(float(episode_reward))
        self._length.append(float(episode_length))
        self._timeout.append(float(timeout))
        if original_reward is not None:
            self._original.append(float(original_reward))

        if steps - self._last_log_step >= self._log_interval:
            self._flush(steps)
            self._last_log_step = steps

    def _flush(self, steps):
        if not self._reward:
            return
        original = float(np.mean(self._original)) if self._original else None
        log_measurement(
            self._log_metric,
            "rollout",
            steps,
            episode_reward=float(np.mean(self._reward)),
            episode_length=float(np.mean(self._length)),
            timeout_rate=float(np.mean(self._timeout)),
            original_reward=original,
        )

    def describe(self):
        """Short pbar fragment with the window-mean reward, or '' when empty."""
        if not self._reward:
            return ""
        return f"rollout_rew : {np.mean(self._reward):8.2f}"
