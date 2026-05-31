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
- ``has_true_reset`` gates the checkpoint-failure ``env.true_reset()`` branch
  (Q-Net only).

The per-episode checkpoint training pulse is shared via
:class:`CheckpointTrainPulse`. Its residual counter is kept on the agent
(through ``read_residual`` / ``write_residual`` accessors) because the DPG
family serializes it as part of the checkpoint state.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


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

        if self._post_pulse is not None:
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
    has_true_reset: Callable[[], bool]
    # agent operations
    train: Callable[[int, int], object]
    evaluate: Callable[[int], object]
    describe: Callable[[object], str]
    bind_loss_window: Callable[[deque], None]
    # checkpoint seam
    checkpoint_on_episode_end: Callable[..., bool]
    checkpoint_pulse: Callable[[int, int], None]


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
        obs = [np.expand_dims(obs, axis=0)]
        lossque = self._begin()
        eval_result = None

        for steps in pbar:
            sel = spec.single_action(obs, steps)
            next_obs, reward, terminated, truncated, info = spec.env.step(sel.env_action)
            next_obs = [np.expand_dims(next_obs, axis=0)]
            spec.replay_buffer.add(obs, sel.store_action, reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = spec.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

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

        for steps in pbar:
            spec.refresh_exploration(steps)
            obs = spec.env.current_obs()
            sel = spec.vector_action(obs, steps)
            spec.env.step(sel.env_action)

            if steps > spec.learning_starts and steps % spec.train_freq == 0:
                for idx in range(spec.worker_size):
                    loss = spec.train(steps + idx, spec.gradient_steps)
                    lossque.append(loss)

            next_obses, rewards, terminateds, truncateds, infos = spec.env.get_result()
            spec.replay_buffer.add(
                [obs], sel.store_action, rewards, [next_obses], terminateds, truncateds
            )

            if steps % spec.eval_freq == 0:
                eval_result = spec.evaluate(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))

    def learn_single_env_checkpointing(self, pbar, callback=None, log_interval=1000, obs=None):
        spec = self.spec
        if obs is None:
            obs, info = spec.env.reset()
            obs = [np.expand_dims(obs, axis=0)]
        lossque = self._begin()
        eval_result = None

        score = 0.0
        eplen = 0

        for steps in pbar:
            eplen += 1
            sel = spec.single_action(obs, steps)
            next_obs, reward, terminated, truncated, info = spec.env.step(sel.env_action)
            next_obs = [np.expand_dims(next_obs, axis=0)]
            spec.replay_buffer.add(obs, sel.store_action, reward, next_obs, terminated, truncated)
            score += float(reward)
            obs = next_obs

            if terminated or truncated:
                if steps > spec.learning_starts:
                    ckpt_success = spec.checkpoint_on_episode_end(
                        steps,
                        score,
                        eplen,
                        train_and_reset_callback=spec.checkpoint_pulse,
                    )
                else:
                    ckpt_success = True
                score = 0.0
                eplen = 0

                if not ckpt_success and spec.has_true_reset():
                    obs, info = spec.env.true_reset()
                else:
                    obs, info = spec.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

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

        for steps in pbar:
            spec.refresh_exploration(steps)
            obs = spec.env.current_obs()
            sel = spec.vector_action(obs, steps)
            spec.env.step(sel.env_action)

            next_obses, rewards, terminateds, truncateds, infos = spec.env.get_result()
            scores += rewards
            eplens += 1

            spec.replay_buffer.add(
                [obs], sel.store_action, rewards, [next_obses], terminateds, truncateds
            )

            if steps > spec.learning_starts:
                done = np.logical_or(terminateds, truncateds)
                done_idx = np.where(done)[0]
                ckpt_results = []
                for idx in done_idx:
                    ckpt_success = spec.checkpoint_on_episode_end(
                        steps,
                        float(scores[idx]),
                        int(eplens[idx]),
                        spec.checkpoint_pulse,
                    )
                    ckpt_results.append((idx, ckpt_success))
                    scores[idx] = 0.0
                    eplens[idx] = 0

                if spec.has_true_reset() and any(not success for _, success in ckpt_results):
                    spec.env.true_reset()

            if steps % spec.eval_freq == 0:
                eval_result = spec.evaluate(steps)

            if steps % log_interval == 0 and eval_result is not None and len(lossque) > 0:
                pbar.set_description(spec.describe(eval_result))
