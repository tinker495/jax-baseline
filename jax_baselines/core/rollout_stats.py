"""Rollout episode statistics for the local off-policy / on-policy families.

The :class:`EpisodeTracker` owns the ``rollout/`` measurement namespace for the
single-process training families. Where ``eval`` produces one aggregated point
per ``eval_freq``, rollout episodes finish at irregular and (vectorized)
parallel times, so completed episodes are pushed into a fixed window and the
window mean is logged periodically.

The tracker writes algorithm-level episode statistics; environment adapters
write their own measurements. It reuses the shared
:func:`jax_baselines.core.eval.log_measurement` tag-writer. The distributed
families keep their own server-side aggregation and do not use this tracker
(documented inconsistency, see ADR 0003).
"""

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.eval import log_measurement
from jax_baselines.core.runtime_adapters import MetricLogger


@dataclass
class TrainingProgress:
    """Performed training transitions and minibatch update rounds for one run.

    Update rounds count a joint actor/critic minibatch once. Time includes
    evaluation within the training session; evaluation transitions are excluded.
    """

    env_steps: int = 0
    update_steps: int = 0
    started_at: float = field(default_factory=perf_counter)

    def log(self, logger: MetricLogger | None, steps: int) -> None:
        if logger is None:
            return
        logger.log_metric("progress/env_steps", self.env_steps, steps)
        logger.log_metric("progress/update_steps", self.update_steps, steps)
        logger.log_metric("time/elapsed_seconds", perf_counter() - self.started_at, steps)


@jax.jit
def device_episode_step(state, rewards, terminateds, truncateds, autoreset):
    """Accumulate one vector step without transferring episode state to the host.

    State is ``(scores, lengths, prev_done)``. Completed rows retain worker order:
    ``(done, score, length, timeout)``.
    """
    scores, lengths, prev_done = state
    active = ~prev_done
    done = (terminateds | truncateds) & active
    scores = scores + jnp.where(active, rewards, 0)
    lengths = lengths + active.astype(lengths.dtype)
    completed = jnp.stack((done, scores, lengths, truncateds), axis=-1)
    return (
        (
            jnp.where(done, 0, scores),
            jnp.where(done, 0, lengths),
            done & autoreset,
        ),
        jnp.where(prev_done, 0, rewards),
        terminateds | prev_done,
        completed,
    )


class EpisodeTracker:
    """Windowed mean of behavior-policy training episodes, logged under ``rollout/``.

    Completed episodes are pushed via :meth:`record`; the window mean is logged
    at most once per ``log_interval`` env steps (throttled on the episode-end
    boundary, so an empty window is never logged). ``K=10`` matches the loss
    ``deque`` convention, trading smoothness for responsiveness.

    The ``log_metric`` callable is bound to the active run and the training
    session releases the tracker before that run's logger leaves scope.
    """

    def __init__(self, log_metric, log_interval, window=10):
        self._log_metric = log_metric
        self._log_interval = log_interval
        self._reward = deque(maxlen=window)
        self._length = deque(maxlen=window)
        self._timeout = deque(maxlen=window)
        self._last_log_step = 0

    def record(self, steps, *, episode_reward, episode_length, timeout):
        """Push one completed episode and flush the window if due.

        ``timeout`` is the per-episode truncation flag (0/1) whose window mean
        is the truncation rate.
        """
        self._reward.append(float(episode_reward))
        self._length.append(float(episode_length))
        self._timeout.append(float(timeout))
        if steps - self._last_log_step >= self._log_interval:
            self._flush(steps)
            self._last_log_step = steps

    def _flush(self, steps):
        if not self._reward:
            return
        log_measurement(
            self._log_metric,
            "rollout",
            steps,
            episode_reward=float(np.mean(self._reward)),
            episode_length=float(np.mean(self._length)),
            timeout_rate=float(np.mean(self._timeout)),
        )

    def describe(self):
        """Short pbar fragment with the window-mean reward, or '' when empty."""
        if not self._reward:
            return ""
        return f"rollout_rew : {np.mean(self._reward):8.2f}"
