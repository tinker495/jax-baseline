"""Local DPG training lifecycle.

Owns replay sampling, SIMBA/reward normalization, PER priority updates, and metric
logging. Algorithm subclasses only provide `_train_state` and the pure update
`_train_step`; the base class compiles single (`_train_on_batch`) and bulk
(`_train_on_bulk`) updates. The host owns the update schedule: every update runs a
compiled variant specialised on static flags (actor update, reset, diagnostics), and
diagnostics are computed only for pulses whose report will be logged. Environment
rollout and the checkpoint training pulse live in `jax_baselines.core.rollout`.
"""

from dataclasses import dataclass

import jax

from jax_baselines.core.bulk_training import (
    bulk_chunk_schedule,
    host_priority_values,
    prepare_replay_batch,
    uses_bulk_pulse,
)


@dataclass
class DPGTrainReport:
    """Update outputs: `loss/qloss` always, diagnostics only on logged pulses."""

    metrics: dict[str, jax.Array]
    metric_counts: dict[str, jax.Array]
    new_priorities: jax.Array | None = None

    def __post_init__(self):
        if self.metric_counts.keys() != self.metrics.keys():
            raise ValueError("Every reported metric needs a count")

    @property
    def loss(self):
        return self.metrics["loss/qloss"]


class DPGTrainingLifecycle:
    """Replay-driven local DPG training lifecycle."""

    def __init__(self, agent):
        self.agent = agent

    def train(self, steps, gradient_steps, logger_run=None, log_interval=None):
        if gradient_steps <= 0:
            raise ValueError("gradient_steps must be greater than 0")

        # One decision per pulse, before running: diagnostics only when the report is logged.
        interval = self.agent.log_interval if log_interval is None else log_interval
        diagnostics = bool(logger_run) and steps - self.agent._last_log_step >= interval
        if self._uses_bulk_pulse(gradient_steps):
            report = self._train_bulk_pulse(gradient_steps, diagnostics)
        else:
            report = self.agent._aggregate_train_reports(
                [self._train_one_batch(diagnostics) for _ in range(gradient_steps)]
            )
        if diagnostics:
            self._log_report(report, steps, logger_run)
        return report.loss

    def _uses_bulk_pulse(self, gradient_steps):
        return uses_bulk_pulse(self.agent, gradient_steps)

    def _train_bulk_pulse(self, gradient_steps, diagnostics):
        """Run chunked bulk updates.

        Bulk mode is a throughput path: one replay sample is split into mini-updates,
        then PER priorities are written back once for the sampled chunk.
        """
        reports = []
        remaining = int(gradient_steps)
        for chunk_size in bulk_chunk_schedule(self.agent, gradient_steps):
            reports.append(self._train_one_bulk_chunk(chunk_size, diagnostics))
            remaining -= chunk_size

        while remaining > 0:
            reports.append(self._train_one_batch(diagnostics))
            remaining -= 1

        return self.agent._aggregate_train_reports(reports)

    # The host train_steps_count schedules the static update flags (and logging/checkpoints);
    # the device counter carried through the compiled update mirrors it for in-jit use.
    def _train_one_bulk_chunk(self, chunk_size, diagnostics):
        plan = self.agent._update_plan(self.agent.train_steps_count + 1, chunk_size, diagnostics)
        self.agent.train_steps_count += chunk_size
        data = self._prepare_batch(
            self._sample_batch(chunk_size * self.agent.batch_size), chunk_size
        )
        report = self.agent._train_on_bulk(data, plan)
        self._update_priorities(data, report)
        return report

    def _train_one_batch(self, diagnostics):
        self.agent.train_steps_count += 1
        flags = self.agent._update_flags(self.agent.train_steps_count, diagnostics)
        data = self._prepare_batch(self._sample_batch())
        report = self.agent._train_on_batch(data, flags)
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

    def _prepare_batch(self, data, chunk_size=None):
        return prepare_replay_batch(
            data,
            chunk_size=chunk_size,
            batch_size=self.agent.batch_size,
            obs_rms=self.agent._policy_update_obs_rms() if self.agent.obs_rms_norm else None,
            rewards=self.agent.reward_normalizer,
        )

    def _update_priorities(self, data, report):
        if not self.agent.prioritized_replay:
            return
        priorities = report.new_priorities
        if not isinstance(data["indexes"], jax.Array):
            priorities = host_priority_values(priorities)
        self.agent.replay_buffer.update_priorities(data["indexes"], priorities)

    def _log_report(self, report, steps, logger_run):
        self.agent._last_log_step = steps
        metrics = report.metrics
        if self.agent.reward_normalizer is not None:
            metrics = {
                **metrics,
                "rollout/reward_scale": self.agent.reward_normalizer.scale,
            }
        metrics, counts = jax.device_get((metrics, report.metric_counts))
        for metric_name, metric_value in metrics.items():
            if metric_name in counts and counts[metric_name] == 0:
                continue
            logger_run.log_metric(metric_name, metric_value, steps)
