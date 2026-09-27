"""Local DPG training lifecycle.

Owns replay sampling, SIMBA/reward normalization, PER priority updates, and metric
logging. Algorithm subclasses only provide `_train_state` and the pure update
`_train_step`; the base class compiles single (`_train_on_batch`) and bulk
(`_train_on_bulk`) updates. The host owns the update schedule: every update runs a
compiled variant specialised on static flags (actor update, reset, diagnostics), and
diagnostics are computed only for pulses whose report will be logged. Environment
rollout and the checkpoint training pulse live in `jax_baselines.core.rollout`.
"""

from dataclasses import dataclass, replace

import jax

from jax_baselines.core.bulk_training import (
    _prepare_batch,
    bulk_chunk_schedule,
    host_priority_values,
    prepare_replay_batch,
    uses_bulk_pulse,
)
from jax_baselines.core.rollout_stats import device_episode_step
from replay_memory.flashbax_buffer import FlashbaxReplayBuffer


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
        self._compiled_replay_update = jax.jit(
            self._update_from_replay, static_argnums=(5, 6), donate_argnums=(1,)
        )
        self._compiled_record_step = jax.jit(
            self._record_device_step, static_argnums=5, donate_argnums=(0, 1)
        )

    def _update_from_replay(
        self, carry, replay_state, replay_key, obs_stats, reward_stats, plan, chunk_size
    ):
        agent = self.agent
        replay = agent.replay_buffer
        replay_key, data = replay._sample(
            replay_state,
            replay_key,
            chunk_size * agent.batch_size,
            agent.prioritized_replay_beta0,
        )
        batch = _prepare_batch(
            {name: value for name, value in data.items() if name != "indexes"},
            obs_stats,
            reward_stats,
            chunk_size=chunk_size,
            batch_size=agent.batch_size,
            flat=False,
            obs_apply=agent.obs_rms.apply if agent.obs_rms_norm else None,
            reward_apply=None if agent.reward_normalizer is None else agent.reward_normalizer.apply,
        )
        carry, outputs = agent._planned_updates(carry, batch, plan)
        if agent.prioritized_replay:
            replay_state = replay._update_priorities(replay_state, data["indexes"], outputs[0])
        return carry, replay_state, replay_key, outputs

    def _record_device_step(
        self,
        replay_state,
        pending,
        episode_state,
        reward_state,
        inputs,
        track_terminations,
    ):
        batch, truncated, active = inputs
        terminated = batch["terminateds"]
        rewards = batch["rewards"]
        if self.agent.reward_normalizer is not None:
            reward_state = self.agent.reward_normalizer.record_state(
                reward_state, (rewards, terminated, truncated, active)
            )
        episode_state, _, _, completed = device_episode_step(
            episode_state,
            rewards,
            terminated if track_terminations else self.agent.replay_buffer._not_truncated,
            truncated if track_terminations else self.agent.replay_buffer._not_truncated,
            self.agent.replay_buffer._not_truncated,
        )
        replay_state, pending = self.agent.replay_buffer._add(
            replay_state, pending, batch, truncated, active
        )
        return replay_state, pending, episode_state, reward_state, completed

    def record_device_step(
        self,
        episode_state,
        obs,
        action,
        rewards,
        next_obs,
        terminated,
        truncated,
        track_terminations,
    ):
        replay = self.agent.replay_buffer
        normalizer = self.agent.reward_normalizer
        if any(
            value.shape != (self.agent.worker_size,) for value in (rewards, terminated, truncated)
        ):
            raise ValueError("Device rollout rewards and done flags must match the worker shape")
        (
            replay.state,
            replay._pending,
            episode_state,
            reward_state,
            completed,
        ) = self._compiled_record_step(
            replay.state,
            replay._pending,
            episode_state,
            None if normalizer is None else normalizer.rollout_state,
            replay.prepare_add(obs, action, rewards, next_obs, terminated, truncated),
            track_terminations,
        )
        if normalizer is not None:
            normalizer.rollout_state = reward_state
        return episode_state, completed

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
            report = self._train_one_bulk_chunk(chunk_size, diagnostics)
            # Priorities are already written back; drop them so a long pulse does not hold them.
            reports.append(replace(report, new_priorities=None))
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
        replay = self.agent.replay_buffer
        # The first sample retains the replay boundary's empty-buffer/argument validation.
        if isinstance(replay, FlashbaxReplayBuffer) and replay._sample_ready:
            obs_rms = self.agent._policy_update_obs_rms() if self.agent.obs_rms_norm else None
            (
                (
                    self.agent._train_state,
                    self.agent._train_key,
                    self.agent._update_count,
                ),
                replay.state,
                replay._key,
                (priorities, metrics, counts),
            ) = self._compiled_replay_update(
                (
                    self.agent._train_state,
                    self.agent._train_key,
                    self.agent._update_count,
                ),
                replay.state,
                replay._key,
                None if obs_rms is None else obs_rms.stats,
                None
                if self.agent.reward_normalizer is None
                else self.agent.reward_normalizer.stats,
                plan,
                chunk_size,
            )
            return DPGTrainReport(metrics, counts, priorities)
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
