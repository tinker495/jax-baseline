"""Local Q-Net training lifecycle.

Centralises replay sampling, PER priority updates, metric/histogram logging,
and checkpoint training pulses behind a single lifecycle object. Algorithms
keep their JIT-compiled tuple returns and translate them into a lifecycle
result on the Python side via `_train_on_batch`.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class QNetTrainContext:
    """Context passed to an algorithm-specific Q-Net train step."""

    steps: int
    train_steps_count: int
    gradient_steps: int


@dataclass
class QNetTrainReport:
    """Normalised metric report from a local Q-Net train step."""

    loss: object
    target: object = None
    metrics: dict = field(default_factory=dict)
    histograms: dict = field(default_factory=dict)

    def __post_init__(self):
        metrics = dict(self.metrics)
        metrics.setdefault("loss/qloss", self.loss)
        if self.target is not None:
            metrics.setdefault("loss/targets", self.target)
        self.metrics = metrics
        self.histograms = dict(self.histograms)


@dataclass(frozen=True)
class QNetTrainResult:
    """Lifecycle-internal train result.

    `report` is the logging/return surface. `replay_priorities` is consumed
    immediately by the lifecycle when PER is enabled and is not part of the
    public metric report.
    """

    report: QNetTrainReport
    replay_priorities: object = None

    @classmethod
    def from_values(
        cls,
        loss,
        target=None,
        replay_priorities=None,
        metrics=None,
        histograms=None,
    ):
        return cls(
            report=QNetTrainReport(
                loss=loss,
                target=target,
                metrics={} if metrics is None else metrics,
                histograms={} if histograms is None else histograms,
            ),
            replay_priorities=replay_priorities,
        )


class QNetTrainingAgentProtocol(Protocol):
    """Private protocol required by `QNetTrainingLifecycle`.

    Q-Net algorithms provide the algorithm-specific `_train_on_batch` hook.
    The base family owns replay sampling, report aggregation, logging cadence,
    and checkpoint counters.
    """

    _qnet_handles_train_pulse: bool
    batch_size: int
    gradient_steps: int
    log_interval: int
    logger_run: object
    prioritized_replay: bool
    prioritized_replay_beta0: float
    replay_buffer: object
    train_freq: int
    train_steps_count: int
    _ckpt_update_residual: int
    _last_log_step: int

    def _sample_batch(self, batch_size=None):
        pass

    def _train_on_batch(self, data, context):
        pass

    def _aggregate_train_reports(self, reports):
        pass


class QNetTrainingLifecycle:
    """Replay-driven local Q-Net training lifecycle."""

    def __init__(self, agent: QNetTrainingAgentProtocol):
        self.agent = agent

    def train(self, steps, gradient_steps):
        if gradient_steps <= 0:
            raise ValueError("gradient_steps must be greater than 0")

        if self.agent._qnet_handles_train_pulse:
            report = self._train_one_pulse(steps, gradient_steps)
        else:
            reports = []
            for _ in range(gradient_steps):
                reports.append(self._train_one_batch(steps, gradient_steps))

            report = self.agent._aggregate_train_reports(reports)

        self._log_report(report, steps)
        return report.loss

    def _train_one_batch(self, steps, gradient_steps):
        self.agent.train_steps_count += 1
        data = self.agent._sample_batch()
        context = QNetTrainContext(
            steps=steps,
            train_steps_count=self.agent.train_steps_count,
            gradient_steps=gradient_steps,
        )
        result = self._normalise_train_result(self.agent._train_on_batch(data, context))
        self._update_priorities(data, result)
        return result.report

    def _train_one_pulse(self, steps, gradient_steps):
        fixed_chunk_size = self.agent.gradient_steps
        if fixed_chunk_size <= 0:
            raise ValueError("agent.gradient_steps must be greater than 0")
        if gradient_steps % fixed_chunk_size != 0:
            raise ValueError(
                "chunked Q-Net training requires gradient_steps to be divisible by "
                f"agent.gradient_steps ({fixed_chunk_size}); got {gradient_steps}"
            )

        num_chunks = gradient_steps // fixed_chunk_size

        reports = []
        for _ in range(num_chunks):
            data = self.agent._sample_batch(fixed_chunk_size * self.agent.batch_size)
            context = QNetTrainContext(
                steps=steps,
                train_steps_count=self.agent.train_steps_count,
                gradient_steps=fixed_chunk_size,
            )
            result = self._normalise_train_result(self.agent._train_on_batch(data, context))
            self.agent.train_steps_count += fixed_chunk_size
            self._update_priorities(data, result)
            reports.append(result.report)

        if len(reports) == 0:
            raise ValueError("gradient_steps must include at least one training chunk")

        return self.agent._aggregate_train_reports(reports)

    def _normalise_train_result(self, result):
        if isinstance(result, QNetTrainResult):
            return result
        if isinstance(result, QNetTrainReport):
            return QNetTrainResult(report=result)
        raise TypeError(
            "_train_on_batch must return QNetTrainResult "
            f"or QNetTrainReport, got {type(result).__name__}"
        )

    def _update_priorities(self, data, result):
        if not self.agent.prioritized_replay:
            return

        if data is None:
            raise ValueError("PER priority update requires sampled replay data")
        if "indexes" not in data:
            raise KeyError("PER priority update requires sampled data to include 'indexes'")
        if result.replay_priorities is None:
            raise ValueError(
                "_train_on_batch must return QNetTrainResult with replay_priorities "
                "when prioritized_replay is enabled"
            )

        self.agent.replay_buffer.update_priorities(data["indexes"], result.replay_priorities)

    def _log_report(self, report, steps):
        logger_run = getattr(self.agent, "logger_run", None)
        if logger_run and (steps - self.agent._last_log_step >= self.agent.log_interval):
            self.agent._last_log_step = steps
            for metric_name, metric_value in report.metrics.items():
                logger_run.log_metric(metric_name, metric_value, steps)
            if hasattr(logger_run, "log_histogram"):
                for histogram_name, histogram_value in report.histograms.items():
                    logger_run.log_histogram(histogram_name, histogram_value, steps)


class QNetRolloutLifecycle:
    """Environment rollout lifecycle for the local Q-Net family.

    This keeps the rollout/checkpoint loop Implementation in the family-local
    lifecycle Module while the base class keeps the public learn_* Interface.
    """

    def __init__(self, agent):
        self.agent = agent

    def learn_single_env(self, pbar, callback=None, log_interval=1000):
        obs, info = self.agent.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.agent.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            actions = self.agent.actions(obs, self.agent.update_eps)
            next_obs, reward, terminated, truncated, info = self.agent.env.step(actions[0][0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.agent.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = self.agent.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.agent.learning_starts and steps % self.agent.train_freq == 0:
                self.agent.update_eps = self.agent.exploration.value(steps)
                loss = self.agent.train_step(steps, self.agent.gradient_steps)
                self.agent.lossque.append(loss)

            if steps % self.agent.eval_freq == 0:
                eval_result = self.agent.eval(steps)

            if (
                steps % log_interval == 0
                and eval_result is not None
                and len(self.agent.lossque) > 0
            ):
                pbar.set_description(self.agent.discription(eval_result))

    def learn_vectorized_env(self, pbar, callback=None, log_interval=1000):
        self.agent.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            self.agent.update_eps = self.agent.exploration.value(steps)
            obs = self.agent.env.current_obs()
            actions = self.agent.actions([obs], self.agent.update_eps)
            self.agent.env.step(actions)

            if steps > self.agent.learning_starts and steps % self.agent.train_freq == 0:
                for idx in range(self.agent.worker_size):
                    loss = self.agent.train_step(steps + idx, self.agent.gradient_steps)
                    self.agent.lossque.append(loss)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.agent.env.get_result()

            self.agent.replay_buffer.add(
                [obs], actions, rewards, [next_obses], terminateds, truncateds
            )

            if steps % self.agent.eval_freq == 0:
                eval_result = self.agent.eval(steps)

            if (
                steps % log_interval == 0
                and eval_result is not None
                and len(self.agent.lossque) > 0
            ):
                pbar.set_description(self.agent.discription(eval_result))

    def learn_single_env_checkpointing(self, pbar, callback=None, log_interval=1000, obs=None):
        if obs is None:
            obs, info = self.agent.env.reset()
            obs = [np.expand_dims(obs, axis=0)]
        self.agent.lossque = deque(maxlen=10)
        eval_result = None

        score = 0.0
        eplen = 0

        for steps in pbar:
            eplen += 1
            actions = self.agent.actions(obs, self.agent.update_eps)
            next_obs, reward, terminated, truncated, info = self.agent.env.step(actions[0][0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.agent.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            score += float(reward)
            obs = next_obs

            if terminated or truncated:
                if steps > self.agent.learning_starts:
                    ckpt_success = self.agent._checkpoint_on_episode_end(
                        steps,
                        score,
                        eplen,
                        train_and_reset_callback=self.agent.checkpointing_adapter.train_and_reset,
                    )
                else:
                    ckpt_success = True
                score = 0.0
                eplen = 0

                if not ckpt_success and self.agent._has_true_reset():
                    obs, info = self.agent.env.true_reset()
                else:
                    obs, info = self.agent.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.agent.learning_starts and steps % self.agent.train_freq == 0:
                self.agent.update_eps = self.agent.exploration.value(steps)

            if steps % self.agent.eval_freq == 0:
                eval_result = self.agent.eval(steps)

            if (
                steps % log_interval == 0
                and eval_result is not None
                and len(self.agent.lossque) > 0
            ):
                pbar.set_description(self.agent.discription(eval_result))

    def learn_vectorized_env_checkpointing(self, pbar, callback=None, log_interval=1000):
        self.agent.lossque = deque(maxlen=10)
        eval_result = None

        scores = np.zeros([self.agent.worker_size], dtype=np.float64)
        eplens = np.zeros([self.agent.worker_size], dtype=np.int32)

        for steps in pbar:
            self.agent.update_eps = self.agent.exploration.value(steps)
            obs = self.agent.env.current_obs()
            actions = self.agent.actions([obs], self.agent.update_eps)
            self.agent.env.step(actions)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.agent.env.get_result()

            scores += rewards
            eplens += 1

            self.agent.replay_buffer.add(
                [obs], actions, rewards, [next_obses], terminateds, truncateds
            )

            if steps > self.agent.learning_starts:
                done = np.logical_or(terminateds, truncateds)
                done_idx = np.where(done)[0]
                ckpt_results = []
                for idx in done_idx:
                    ckpt_success = self.agent._checkpoint_on_episode_end(
                        steps,
                        float(scores[idx]),
                        int(eplens[idx]),
                        self.agent.checkpointing_adapter.train_and_reset,
                    )
                    ckpt_results.append((idx, ckpt_success))
                    scores[idx] = 0.0
                    eplens[idx] = 0

                if self.agent._has_true_reset() and any(not success for _, success in ckpt_results):
                    try:
                        self.agent.env.true_reset()
                    except Exception:
                        self.agent.env.reset()

            if steps % self.agent.eval_freq == 0:
                eval_result = self.agent.eval(steps)

            if (
                steps % log_interval == 0
                and eval_result is not None
                and len(self.agent.lossque) > 0
            ):
                pbar.set_description(self.agent.discription(eval_result))


class QNetCheckpointingAdapter:
    """Checkpoint training-pulse adapter for the local Q-Net lifecycle."""

    def __init__(self, agent):
        self.agent = agent

    def train_and_reset(self, step_val, accumulated_timesteps):
        self.agent._ckpt_update_residual += int(accumulated_timesteps)
        num_update_iters = 0
        while self.agent._ckpt_update_residual >= self.agent.train_freq:
            self.agent._ckpt_update_residual -= self.agent.train_freq
            num_update_iters += 1

        if num_update_iters > 0:
            total_updates = num_update_iters * self.agent.gradient_steps
            loss = self.agent.train_step(step_val, total_updates)
            self.agent.lossque.append(loss)
