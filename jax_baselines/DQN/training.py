"""Local Q-Net training lifecycle.

Centralises replay sampling, PER priority updates, and metric/histogram logging
behind a single training-lifecycle object. Algorithms keep their JIT-compiled
tuple returns and translate them into a lifecycle result on the Python side via
`_train_on_batch`. Environment rollout and the checkpoint training pulse live in
`jax_baselines.core.rollout`.
"""

from dataclasses import dataclass, field
from typing import Protocol


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
    The base family owns replay sampling, report aggregation, and logging
    cadence.
    """

    _qnet_handles_train_pulse: bool
    batch_size: int
    gradient_steps: int
    log_interval: int
    logger_run: object
    prioritized_replay: bool
    prioritized_replay_beta0: float
    replay_buffer: object
    train_steps_count: int
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
        logger_run = self.agent.logger_run
        if logger_run and (steps - self.agent._last_log_step >= self.agent.log_interval):
            self.agent._last_log_step = steps
            for metric_name, metric_value in report.metrics.items():
                logger_run.log_metric(metric_name, metric_value, steps)
            if hasattr(logger_run, "log_histogram"):
                for histogram_name, histogram_value in report.histograms.items():
                    logger_run.log_histogram(histogram_name, histogram_value, steps)
