"""Local DPG training lifecycle.

Owns replay sampling, SIMBA normalization, PER priority updates, and metric
logging. Algorithm subclasses only provide the per-batch gradient update via
`_train_on_batch`. Environment rollout and the checkpoint training pulse live
in `jax_baselines.core.rollout`.
"""

from dataclasses import dataclass, field
from typing import Protocol

from jax_baselines.core.bulk_training import (
    bulk_chunk_schedule,
    bulk_train_hook,
    flatten_priority_values,
    make_train_contexts,
    normalize_bulk_weights,
    reshape_bulk_batch,
    uses_bulk_pulse,
)


@dataclass(frozen=True)
class DPGTrainContext:
    """Context passed to `_train_on_batch`."""

    steps: int
    train_steps_count: int


@dataclass
class DPGTrainReport:
    """Normalized report returned by `_train_on_batch`."""

    loss: object
    target: object = None
    new_priorities: object = None
    metrics: dict = field(default_factory=dict)
    update_count: int = 1

    def __post_init__(self):
        metrics = dict(self.metrics)
        metrics.setdefault("loss/qloss", self.loss)
        if self.target is not None:
            metrics.setdefault("loss/targets", self.target)
        self.metrics = metrics


class DPGTrainingAgentProtocol(Protocol):
    batch_size: int
    log_interval: int
    logger_run: object
    max_bulk_updates_per_pulse: int
    prioritized_replay: bool
    prioritized_replay_beta0: float
    replay_buffer: object
    simba: bool
    supports_bulk_training: bool
    train_steps_count: int
    _last_log_step: int

    def _aggregate_train_reports(self, reports):
        pass

    def _policy_update_obs_rms(self):
        pass

    def _train_on_batch(self, data, context):
        pass

    def _train_on_bulk(self, data, contexts):
        pass


class DPGTrainingLifecycle:
    """Replay-driven local DPG training lifecycle."""

    def __init__(self, agent: DPGTrainingAgentProtocol):
        self.agent = agent

    def train(self, steps, gradient_steps):
        if gradient_steps <= 0:
            raise ValueError("gradient_steps must be greater than 0")

        if self._uses_bulk_pulse(gradient_steps):
            report = self._train_bulk_pulse(steps, gradient_steps)
            self._log_report(report, steps)
            return report.loss

        reports = []
        for _ in range(gradient_steps):
            reports.append(self._train_one_batch(steps))

        report = self.agent._aggregate_train_reports(reports)
        self._log_report(report, steps)
        return report.loss

    def _uses_bulk_pulse(self, gradient_steps):
        return uses_bulk_pulse(self.agent, gradient_steps)

    def _train_bulk_pulse(self, steps, gradient_steps):
        """Run chunked bulk updates.

        Bulk mode is a throughput path: one replay sample is split into mini-updates,
        then PER priorities are written back once for the sampled chunk.
        """
        train_on_bulk = bulk_train_hook(self.agent)

        reports = []
        remaining = int(gradient_steps)
        for chunk_size in bulk_chunk_schedule(self.agent, gradient_steps):
            reports.append(self._train_one_bulk_chunk(steps, chunk_size, train_on_bulk))
            remaining -= chunk_size

        while remaining > 0:
            reports.append(self._train_one_batch(steps))
            remaining -= 1

        return self.agent._aggregate_train_reports(reports)

    def _train_one_bulk_chunk(self, steps, chunk_size, train_on_bulk):
        contexts = make_train_contexts(self.agent, DPGTrainContext, steps, chunk_size)

        data = self._sample_batch(chunk_size * self.agent.batch_size)
        data = self._reshape_bulk_batch(data, chunk_size)
        data = normalize_bulk_weights(data)
        self._normalize_batch(data)
        report = train_on_bulk(data, contexts)
        self._update_priorities(data, report)
        return report

    def _train_one_batch(self, steps):
        self.agent.train_steps_count += 1
        data = self._sample_batch()
        self._normalize_batch(data)
        context = DPGTrainContext(
            steps=steps,
            train_steps_count=self.agent.train_steps_count,
        )
        report = self.agent._train_on_batch(data, context)
        self._update_priorities(data, report)
        return report

    def _sample_batch(self, batch_size=None):
        batch_size = self.agent.batch_size if batch_size is None else batch_size
        if self.agent.prioritized_replay:
            return self.agent.replay_buffer.sample(
                batch_size,
                self.agent.prioritized_replay_beta0,
            )
        return self.agent.replay_buffer.sample(batch_size)

    def _reshape_bulk_batch(self, data, chunk_size):
        return reshape_bulk_batch(data, chunk_size, self.agent.batch_size)

    def _normalize_batch(self, data):
        if self.agent.simba:
            obs_rms = self.agent._policy_update_obs_rms()
            data["obses"] = obs_rms.normalize(data["obses"])
            data["nxtobses"] = obs_rms.normalize(data["nxtobses"])

    def _update_priorities(self, data, report):
        if self.agent.prioritized_replay:
            indexes = flatten_priority_values(data["indexes"])
            priorities = flatten_priority_values(report.new_priorities)
            self.agent.replay_buffer.update_priorities(indexes, priorities)

    def _log_report(self, report, steps):
        logger_run = self.agent.logger_run
        if logger_run and (steps - self.agent._last_log_step >= self.agent.log_interval):
            self.agent._last_log_step = steps
            for metric_name, metric_value in report.metrics.items():
                logger_run.log_metric(metric_name, metric_value, steps)
