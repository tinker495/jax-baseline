"""Local DPG training lifecycle Interface.

This Module keeps replay sampling, SIMBA normalization, PER priority updates,
metric logging, and checkpoint training pulses behind one lifecycle Interface.
Algorithm Implementations only provide the per-batch gradient update.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


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


class DPGRolloutLifecycle:
    """Environment rollout lifecycle for the local DPG family.

    This Module keeps DPG rollout/checkpoint loop Implementation family-local
    and lets the base class expose thin learn_* Interface methods.
    """

    def __init__(self, agent):
        self.agent = agent

    def learn_single_env(self, pbar, callback=None, log_interval=1000):
        obs, info = self.agent.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.agent.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            actions = self.agent.actions(obs, steps)
            next_obs, reward, terminated, truncated, info = self.agent.env.step(actions[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.agent.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = self.agent.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.agent.learning_starts and steps % self.agent.train_freq == 0:
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
            obs = self.agent.env.current_obs()
            actions = self.agent.actions([obs], steps)
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
            actions = self.agent.actions(obs, steps)
            next_obs, reward, terminated, truncated, info = self.agent.env.step(actions[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.agent.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            score += float(reward)
            obs = next_obs

            if terminated or truncated:
                if steps > self.agent.learning_starts:
                    self.agent._checkpoint_on_episode_end(
                        steps,
                        score,
                        eplen,
                        train_and_reset_callback=self.agent.checkpointing_adapter.train_and_reset,
                    )
                score = 0.0
                eplen = 0
                obs, info = self.agent.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

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
            obs = self.agent.env.current_obs()
            actions = self.agent.actions([obs], steps)
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
                done_mask = np.logical_or(terminateds, truncateds)
                done_indices = np.where(done_mask)[0]
                for idx in done_indices:
                    self.agent._checkpoint_on_episode_end(
                        steps,
                        float(scores[idx]),
                        int(eplens[idx]),
                        self.agent.checkpointing_adapter.train_and_reset,
                    )
                    scores[idx] = 0.0
                    eplens[idx] = 0

            if steps % self.agent.eval_freq == 0:
                eval_result = self.agent.eval(steps)

            if (
                steps % log_interval == 0
                and eval_result is not None
                and len(self.agent.lossque) > 0
            ):
                pbar.set_description(self.agent.discription(eval_result))


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
