"""Unified environment-rollout engine for the off-policy local training families.

Hosts the four ``learn_*`` loops shared by the Q-Net and DPG families behind a
narrow :class:`RolloutSpec` contract, replacing the per-family
``*RolloutLifecycle`` / ``*CheckpointingAdapter`` collaborators that previously
duplicated this control flow.

The engine never touches a training agent directly: every interaction goes
through the injected ``RolloutSpec``. Family-specific behavior is supplied as
spec callbacks:

- ``single_action`` / ``vector_action`` produce an :class:`ActionSelection`
  (the Q-Net family double-indexes discrete actions, the DPG family does not);
- ``refresh_exploration`` updates epsilon for the Q-Net family and is a no-op
  for the DPG family;
- ``force_reset`` optionally supplies the adapter's checkpoint-failure reset.

The per-episode checkpoint training pulse is shared via
:class:`CheckpointTrainPulse`. Its residual counter is kept on the agent
(through ``read_residual`` / ``write_residual`` accessors) because the DPG
family serializes it as part of the checkpoint state.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from jax_baselines.core.env_protocols import (
    batch_observation,
    single_real_episode_end,
    vector_autoreset_mask,
    vector_real_reset_mask,
)
from jax_baselines.core.eval import (
    extract_original_reward,
    extract_vector_original_rewards,
)


@dataclass(frozen=True)
class ActionSelection:
    """One rollout step's action, split into the env-step and stored forms.

    ``env_action`` is passed to ``env.step``; ``store_action`` is the action
    written to the replay buffer. They differ for the Q-Net family in
    single-env loops (discrete actions are double-indexed for ``env.step``).
    """

    env_action: object
    store_action: object


class CheckpointTrainPulse:
    """Episode-boundary training pulse shared by the local families.

    Converts accumulated episode timesteps into ``train_freq``-aligned gradient
    pulses (TD7-style checkpoint training). The residual counter lives on the
    agent and is reached through ``read_residual`` / ``write_residual`` because
    the DPG family persists it in its serialized checkpoint state.
    """

    def __init__(
        self,
        *,
        train_freq: int,
        gradient_steps: int,
        train: Callable[[int, int], object],
        record_loss: Callable[[object], None],
        read_residual: Callable[[], int],
        write_residual: Callable[[int], None],
        post_pulse: Optional[Callable[[], None]] = None,
    ):
        self._train_freq = train_freq
        self._gradient_steps = gradient_steps
        self._train = train
        self._record_loss = record_loss
        self._read_residual = read_residual
        self._write_residual = write_residual
        self._post_pulse = post_pulse

    def __call__(self, steps, accumulated_timesteps):
        residual = self._read_residual() + int(accumulated_timesteps)
        num_update_iters = 0
        while residual >= self._train_freq:
            residual -= self._train_freq
            num_update_iters += 1
        self._write_residual(residual)

        if num_update_iters > 0:
            loss = self._train(steps, num_update_iters * self._gradient_steps)
            self._record_loss(loss)

        if num_update_iters > 0 and self._post_pulse is not None:
            self._post_pulse()


@dataclass
class RolloutSpec:
    """Narrow contract the :class:`RolloutEngine` needs from a training agent."""

    env: object
    replay_buffer: object
    learning_starts: int
    train_freq: int
    gradient_steps: int
    eval_freq: int
    worker_size: int
    # policy seam (family-specific)
    single_action: Callable[[object, int], ActionSelection]
    vector_action: Callable[[object, int], ActionSelection]
    refresh_exploration: Callable[[int], None]
    force_reset: Callable[[], object] | None
    # agent operations
    train: Callable[[int, int], object]
    evaluate: Callable[[int], object]
    describe: Callable[[object], str]
    bind_loss_window: Callable[[deque], None]
    # rollout-measurement seam: hands each completed training episode to the
    # agent's EpisodeTracker (the engine stays logger-free).
    record_rollout_episode: Callable[..., None]
    # checkpoint seam
    checkpoint_on_episode_end: Callable[..., bool]
    checkpoint_pulse: Callable[[int, int], None]
    checkpoint_monitor_worker: int = 0
    reward_normalization: bool = False
    record_transition: Callable[..., None] | None = None


class RolloutEngine:
    """Replay-driven environment rollout loops for the local off-policy families."""

    def __init__(self, spec: RolloutSpec):
        self.spec = spec

    def _begin(self):
        lossque = deque(maxlen=10)
        self.spec.bind_loss_window(lossque)
        return lossque

    def learn_single_env(self, pbar, callback=None, log_interval=1000):
        spec = self.spec
        obs, info = spec.env.reset()
        obs = batch_observation(obs)
        lossque = self._begin()
        eval_result = None

        score = 0.0
        eplen = 0
        original = 0.0
        have_original = False

        for steps in pbar:
            sel = spec.single_action(obs, steps)
            next_obs, reward, terminated, truncated, info = spec.env.step(sel.env_action)
            next_obs = batch_observation(next_obs)
            if spec.reward_normalization and spec.record_transition is not None:
                spec.record_transition(reward, np.logical_or(terminated, truncated))
            spec.replay_buffer.add(obs, sel.store_action, reward, next_obs, terminated, truncated)
            score += float(reward)
            eplen += 1
            step_original = extract_original_reward(info)
            if step_original is not None:
                have_original = True
                original += float(step_original)
            obs = next_obs

            if terminated or truncated:
                emit_original = have_original and single_real_episode_end(
                    terminated, truncated, info
                )
                spec.record_rollout_episode(
                    steps,
                    episode_reward=score,
                    episode_length=eplen,
                    timeout=float(truncated),
                    original_reward=original if emit_original else None,
                )
                score = 0.0
                eplen = 0
                if emit_original:
                    original = 0.0
                    have_original = False
                obs, info = spec.env.reset()
                obs = batch_observation(obs)

            if steps > spec.learning_starts and steps % spec.train_freq == 0:
                spec.refresh_exploration(steps)
                loss = spec.train(steps, spec.gradient_steps)
                lossque.append(loss)

            if steps % spec.eval_freq == 0:
                eval_result = spec.evaluate(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))

    def learn_vectorized_env(self, pbar, callback=None, log_interval=1000):
        spec = self.spec
        lossque = self._begin()
        eval_result = None
        scores = np.zeros([spec.worker_size], dtype=np.float64)
        eplens = np.zeros([spec.worker_size], dtype=np.int32)
        originals = np.zeros([spec.worker_size], dtype=np.float64)
        original_present = np.zeros([spec.worker_size], dtype=bool)
        # Adapters classify two backend quirks separately: whether this done row
        # is a real game reset, and whether the next row is an autoreset dummy.
        prev_done = None
        train_residual = 0

        for steps in pbar:
            spec.refresh_exploration(steps)
            obs = spec.env.current_obs()
            sel = spec.vector_action(obs, steps)
            spec.env.step(sel.env_action)

            next_obses, rewards, terminateds, truncateds, infos = spec.env.get_result()
            done = np.logical_or(terminateds, truncateds)
            active = np.ones(spec.worker_size, dtype=bool) if prev_done is None else ~prev_done
            if spec.reward_normalization and spec.record_transition is not None:
                spec.record_transition(rewards, done, active)
            scores[active] += rewards[active]
            eplens[active] += 1
            step_original, step_original_present = extract_vector_original_rewards(
                infos, spec.worker_size
            )
            real_reset = vector_real_reset_mask(spec.env, terminateds, truncateds, infos)
            autoreset = vector_autoreset_mask(spec.env, terminateds, truncateds, infos)
            active_original = active & step_original_present
            originals[active_original] += step_original[active_original]
            original_present[active_original] = True

            if steps > spec.learning_starts:
                train_residual += int(active.sum())
                update_iters, train_residual = divmod(train_residual, spec.train_freq)
                if update_iters > 0:
                    loss = spec.train(steps, update_iters * spec.gradient_steps)
                    lossque.append(loss)

            store_mask = None if prev_done is None or not prev_done.any() else ~prev_done
            spec.replay_buffer.add(
                obs,
                sel.store_action,
                rewards,
                next_obses,
                terminateds,
                truncateds,
                store_mask=store_mask,
            )

            for idx in np.where(done & active)[0]:
                emit_original = original_present[idx] and real_reset[idx]
                spec.record_rollout_episode(
                    steps,
                    episode_reward=float(scores[idx]),
                    episode_length=int(eplens[idx]),
                    timeout=float(truncateds[idx]),
                    original_reward=float(originals[idx]) if emit_original else None,
                )
                scores[idx] = 0.0
                eplens[idx] = 0
                if emit_original:
                    originals[idx] = 0.0
                    original_present[idx] = False

            prev_done = done & autoreset & active

            if steps % spec.eval_freq == 0:
                eval_result = spec.evaluate(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))

    def learn_single_env_checkpointing(self, pbar, callback=None, log_interval=1000, obs=None):
        spec = self.spec
        if obs is None:
            obs, info = spec.env.reset()
            obs = batch_observation(obs)
        lossque = self._begin()
        eval_result = None

        score = 0.0
        eplen = 0
        original = 0.0
        have_original = False

        for steps in pbar:
            eplen += 1
            sel = spec.single_action(obs, steps)
            next_obs, reward, terminated, truncated, info = spec.env.step(sel.env_action)
            next_obs = batch_observation(next_obs)
            if spec.reward_normalization and spec.record_transition is not None:
                spec.record_transition(reward, np.logical_or(terminated, truncated))
            spec.replay_buffer.add(obs, sel.store_action, reward, next_obs, terminated, truncated)
            score += float(reward)
            step_original = extract_original_reward(info)
            if step_original is not None:
                have_original = True
                original += float(step_original)
            obs = next_obs

            if terminated or truncated:
                emit_original = have_original and single_real_episode_end(
                    terminated, truncated, info
                )
                if steps > spec.learning_starts:
                    ckpt_success = spec.checkpoint_on_episode_end(
                        steps,
                        score,
                        eplen,
                        train_and_reset_callback=spec.checkpoint_pulse,
                    )
                    spec.record_rollout_episode(
                        steps,
                        episode_reward=score,
                        episode_length=eplen,
                        timeout=float(truncated),
                        original_reward=original if emit_original else None,
                    )
                else:
                    ckpt_success = True
                score = 0.0
                eplen = 0
                if emit_original:
                    original = 0.0
                    have_original = False

                if not ckpt_success and spec.force_reset is not None:
                    obs, info = spec.force_reset()
                else:
                    obs, info = spec.env.reset()
                obs = batch_observation(obs)

            if steps > spec.learning_starts and steps % spec.train_freq == 0:
                spec.refresh_exploration(steps)

            if steps % spec.eval_freq == 0:
                eval_result = spec.evaluate(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))

    def learn_vectorized_env_checkpointing(self, pbar, callback=None, log_interval=1000):
        spec = self.spec
        lossque = self._begin()
        eval_result = None

        scores = np.zeros([spec.worker_size], dtype=np.float64)
        eplens = np.zeros([spec.worker_size], dtype=np.int32)
        originals = np.zeros([spec.worker_size], dtype=np.float64)
        original_present = np.zeros([spec.worker_size], dtype=bool)
        # Adapters classify two backend quirks separately: whether this done row
        # is a real game reset, and whether the next row is an autoreset dummy.
        prev_done = None
        defer_checkpoint_pulses = spec.force_reset is None
        pending_checkpoint_pulses = deque()
        pending_eval_steps = deque()

        def checkpoint_pulse_callback(pulse_steps, accumulated_timesteps):
            if defer_checkpoint_pulses:
                pending_checkpoint_pulses.append((pulse_steps, accumulated_timesteps))
            else:
                spec.checkpoint_pulse(pulse_steps, accumulated_timesteps)

        def run_eval(eval_steps):
            nonlocal eval_result
            eval_result = spec.evaluate(eval_steps)

        def schedule_eval(eval_steps):
            if pending_checkpoint_pulses:
                pending_eval_steps.append(eval_steps)
            else:
                run_eval(eval_steps)

        def flush_checkpoint_pulses():
            while pending_checkpoint_pulses:
                pulse_steps, accumulated_timesteps = pending_checkpoint_pulses.popleft()
                spec.checkpoint_pulse(pulse_steps, accumulated_timesteps)
            while pending_eval_steps:
                run_eval(pending_eval_steps.popleft())

        for steps in pbar:
            spec.refresh_exploration(steps)
            obs = spec.env.current_obs()
            sel = spec.vector_action(obs, steps)
            spec.env.step(sel.env_action)
            flush_checkpoint_pulses()

            next_obses, rewards, terminateds, truncateds, infos = spec.env.get_result()
            done = np.logical_or(terminateds, truncateds)
            active = np.ones(spec.worker_size, dtype=bool) if prev_done is None else ~prev_done
            if spec.reward_normalization and spec.record_transition is not None:
                spec.record_transition(rewards, done, active)
            scores[active] += rewards[active]
            eplens[active] += 1
            step_original, step_original_present = extract_vector_original_rewards(
                infos, spec.worker_size
            )
            real_reset = vector_real_reset_mask(spec.env, terminateds, truncateds, infos)
            autoreset = vector_autoreset_mask(spec.env, terminateds, truncateds, infos)
            active_original = active & step_original_present
            originals[active_original] += step_original[active_original]
            original_present[active_original] = True

            store_mask = None if prev_done is None or not prev_done.any() else ~prev_done
            spec.replay_buffer.add(
                obs,
                sel.store_action,
                rewards,
                next_obses,
                terminateds,
                truncateds,
                store_mask=store_mask,
            )
            if steps > spec.learning_starts:
                done_idx = np.where(done & active)[0]
                monitor_worker = spec.checkpoint_monitor_worker
                # Only the monitor worker advances the checkpoint criterion, so the
                # assessment sees a clean single-policy episode stream; every
                # finished episode still feeds its timesteps into the pulse volume.
                monitor_failed = False
                checkpoint_order = [idx for idx in done_idx if idx != monitor_worker]
                if monitor_worker in done_idx:
                    checkpoint_order.append(monitor_worker)
                for idx in checkpoint_order:
                    advance = idx == monitor_worker
                    ckpt_success = spec.checkpoint_on_episode_end(
                        steps,
                        float(scores[idx]),
                        int(eplens[idx]),
                        checkpoint_pulse_callback,
                        advance_criterion=advance,
                    )
                    if advance and not ckpt_success:
                        monitor_failed = True
                    emit_original = original_present[idx] and real_reset[idx]
                    if active[idx]:
                        spec.record_rollout_episode(
                            steps,
                            episode_reward=float(scores[idx]),
                            episode_length=int(eplens[idx]),
                            timeout=float(truncateds[idx]),
                            original_reward=(float(originals[idx]) if emit_original else None),
                        )
                    scores[idx] = 0.0
                    eplens[idx] = 0
                    if active[idx] and emit_original:
                        originals[idx] = 0.0
                        original_present[idx] = False

                if spec.force_reset is not None and monitor_failed:
                    spec.force_reset()

            prev_done = done & autoreset & active

            if steps % spec.eval_freq == 0:
                schedule_eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))

        flush_checkpoint_pulses()
