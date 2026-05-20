"""Local DPG training lifecycle Interface.

This Module keeps replay sampling, SIMBA normalization, PER priority updates,
metric logging, and checkpoint training pulses behind one lifecycle Interface.
Algorithm Implementations only provide the per-batch gradient update.
"""

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DPGTrainContext:
    """Context passed to an algorithm-specific DPG train Implementation."""

    steps: int
    train_steps_count: int


@dataclass
class DPGTrainReport:
    """Normalized report returned by a local DPG train Implementation."""

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
    """Replay-driven local DPG training lifecycle Module."""

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
        logger_run = getattr(self.agent, "logger_run", None)
        if logger_run and (steps - self.agent._last_log_step >= self.agent.log_interval):
            self.agent._last_log_step = steps
            for metric_name, metric_value in report.metrics.items():
                logger_run.log_metric(metric_name, metric_value, steps)


class DPGCheckpointingAdapter:
    """Checkpoint training-pulse Adapter for the local DPG lifecycle."""

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
            # Keep the public train_step Interface as the seam so excluded legacy
            # Implementations can keep their overrides while local DPG variants
            # inherit the unified lifecycle.
            loss = self.agent.train_step(step_val, total_updates)
            self.agent.lossque.append(loss)

        self._snapshot_action_normalizer()

    def _snapshot_action_normalizer(self):
        if getattr(self.agent, "simba", False) and hasattr(self.agent, "obs_rms"):
            try:
                self.agent.action_obs_rms = deepcopy(self.agent.obs_rms)
            except Exception:
                pass
