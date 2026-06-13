"""Local DPG training lifecycle.

Owns replay sampling, SIMBA normalization, PER priority updates, and metric
logging. Algorithm subclasses only provide the per-batch gradient update via
`_train_on_batch`. Environment rollout and the checkpoint training pulse live
in `jax_baselines.common.rollout`.
"""

from dataclasses import dataclass, field


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

    def __post_init__(self):
        metrics = dict(self.metrics)
        metrics.setdefault("loss/qloss", self.loss)
        if self.target is not None:
            metrics.setdefault("loss/targets", self.target)
        self.metrics = metrics


class DPGTrainingLifecycle:
    """Replay-driven local DPG training lifecycle."""

    def __init__(self, agent):
        self.agent = agent

    def train(self, steps, gradient_steps):
        reports = []
        for _ in range(gradient_steps):
            reports.append(self._train_one_batch(steps))

        if len(reports) == 0:
            raise ValueError("gradient_steps must be greater than 0")

        report = self.agent._aggregate_train_reports(reports)
        self._log_report(report, steps)
        return report.loss

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

    def _sample_batch(self):
        if self.agent.prioritized_replay:
            return self.agent.replay_buffer.sample(
                self.agent.batch_size,
                self.agent.prioritized_replay_beta0,
            )
        return self.agent.replay_buffer.sample(self.agent.batch_size)

    def _normalize_batch(self, data):
        if self.agent.simba:
            data["obses"] = self.agent.obs_rms.normalize(data["obses"])
            data["nxtobses"] = self.agent.obs_rms.normalize(data["nxtobses"])

    def _update_priorities(self, data, report):
        if self.agent.prioritized_replay:
            self.agent.replay_buffer.update_priorities(data["indexes"], report.new_priorities)

    def _log_report(self, report, steps):
        logger_run = self.agent.logger_run
        if logger_run and (steps - self.agent._last_log_step >= self.agent.log_interval):
            self.agent._last_log_step = steps
            for metric_name, metric_value in report.metrics.items():
                logger_run.log_metric(metric_name, metric_value, steps)
