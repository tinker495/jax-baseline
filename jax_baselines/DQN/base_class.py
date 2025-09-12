from collections import deque
from copy import deepcopy

import jax
import numpy as np
from tqdm.auto import trange

from jax_baselines.common.env_info import get_local_env_info
from jax_baselines.common.eval import evaluate_policy, record_and_test
from jax_baselines.common.logger import TensorboardLogger
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.replay_factory import make_replay_buffer
from jax_baselines.common.schedules import ConstantSchedule, LinearSchedule
from jax_baselines.common.utils import compute_ckpt_window_stat, key_gen, restore, save


class Q_Network_Family(object):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        double_q=False,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        param_noise=False,
        munchausen=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
        # Checkpointing options (opt-in by default for base class)
        use_checkpointing=True,
        steps_before_checkpointing=500000,
        max_eps_before_checkpointing=10,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="median",
        ckpt_baseline_q=None,
        ckpt_gate_mode=None,
        ckpt_gate_q=None,
    ):
        self.name = "Q_Network_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = int(
            np.ceil(target_network_update_freq / train_freq) * train_freq
        )
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._gamma = np.power(gamma, n_step)  # n_step gamma
        self.log_dir = log_dir
        self.double_q = double_q
        self.dueling_model = dueling_model
        self.n_step_method = n_step > 1
        self.n_step = n_step
        self.munchausen = munchausen
        self.munchausen_alpha = 0.9
        self.munchausen_entropy_tau = 0.03

        self.train_steps_count = 0
        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-3 / self.batch_size)
        self.optimizer_name = optimizer

        self.compress_memory = compress_memory

        self.get_env_setup()
        self.get_memory_setup()

        # Generic checkpointing scaffolding (used by algorithms that opt-in)
        self.use_checkpointing = use_checkpointing
        self.checkpointing_enabled = False  # becomes True after steps_before_checkpointing
        self.steps_before_checkpointing = min(int(steps_before_checkpointing), learning_starts * 2)
        self.max_eps_before_checkpointing = int(max_eps_before_checkpointing)
        self.initial_checkpoint_window = int(initial_checkpoint_window)
        self._ckpt_eps_since_update = 0
        self._ckpt_timesteps_since_update = 0
        self._ckpt_max_eps_before_update = self.initial_checkpoint_window

        # Robust checkpointing controls
        self.ckpt_quantile = 0.2  # q-quantile statistic instead of strict min
        self.use_ckpt_return_standardization = False  # compare windows in absolute return space
        self._ckpt_returns_window = []  # recent episode returns in current window
        self._ckpt_baseline = -1e8  # window-stat baseline (initialized on first window)
        self._ckpt_update_residual = 0  # exact training-parity residual accumulator
        # Track snapshot updates for logging and progress description
        self._last_ckpt_update_step = None
        self._ckpt_update_count = 0

        # Checkpoint baseline mode configuration
        self.ckpt_baseline_mode = ckpt_baseline_mode
        self.ckpt_baseline_q = (
            ckpt_baseline_q if ckpt_baseline_q is not None else self.ckpt_quantile
        )

        # Checkpoint gating mode configuration
        self.ckpt_gate_mode = (
            ckpt_gate_mode if ckpt_gate_mode is not None else self.ckpt_baseline_mode
        )
        if ckpt_gate_q is not None:
            self.ckpt_gate_q = ckpt_gate_q
        elif self.ckpt_gate_mode == "median":
            self.ckpt_gate_q = 0.5
        elif self.ckpt_gate_mode in ["quantile", "min", "mean"]:
            self.ckpt_gate_q = self.ckpt_baseline_q
        else:
            self.ckpt_gate_q = self.ckpt_quantile

        # Logging throttle based on last log step
        self._last_log_step = 0

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            # Calls overridden setup_model in children
            self.setup_model()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        (
            self.env,
            self.eval_env,
            self.observation_space,
            self.action_size,
            self.worker_size,
            self.env_type,
        ) = get_local_env_info(self.env_builder, self.num_workers)
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")

    def get_memory_setup(self):
        # Use factory to select correct buffer implementation
        self.replay_buffer = make_replay_buffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_shape_or_n=1,
            worker_size=self.worker_size,
            n_step=self.n_step if self.n_step_method else 1,
            gamma=self.gamma,
            prioritized=self.prioritized_replay,
            alpha=self.prioritized_replay_alpha,
            eps=self.prioritized_replay_eps,
            compress_memory=self.compress_memory,
        )

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def _compile_common_functions(self):
        """Common JIT compilation for Q-Network family algorithms."""
        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._loss = jax.jit(self._loss)
        self._target = jax.jit(self._target)
        self._train_step = jax.jit(self._train_step)

    def _sample_batch(self, batch_size=None):
        """Common batch sampling logic for Q-Network family algorithms."""
        if batch_size is None:
            batch_size = self.batch_size
        if self.prioritized_replay:
            return self.replay_buffer.sample(batch_size, self.prioritized_replay_beta0)
        else:
            return self.replay_buffer.sample(batch_size)

    def _update_priorities(self, data, new_priorities):
        """Common priority update logic for Q-Network family algorithms."""
        if self.prioritized_replay:
            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

    def _common_train_step_wrapper(self, steps, gradient_steps, train_step_func):
        """Common training step wrapper for Q-Network family algorithms."""
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            data = self._sample_batch()

            result = train_step_func(data)

            # Extract results based on the algorithm
            if (
                len(result) == 6
            ):  # DQN style: params, target_params, opt_state, loss, t_mean, new_priorities
                (
                    self.params,
                    self.target_params,
                    self.opt_state,
                    loss,
                    t_mean,
                    new_priorities,
                ) = result
                self._update_priorities(data, new_priorities)

                if self.logger_run and (steps - self._last_log_step >= self.log_interval):
                    self._last_log_step = steps
                    self.logger_run.log_metric("loss/qloss", loss, steps)
                    self.logger_run.log_metric("loss/targets", t_mean, steps)

            elif (
                len(result) == 7
            ):  # QRDQN/IQN style: params, target_params, opt_state, loss, t_mean, t_std, new_priorities
                (
                    self.params,
                    self.target_params,
                    self.opt_state,
                    loss,
                    t_mean,
                    t_std,
                    new_priorities,
                ) = result
                self._update_priorities(data, new_priorities)

                if self.logger_run and (steps - self._last_log_step >= self.log_interval):
                    self._last_log_step = steps
                    self.logger_run.log_metric("loss/qloss", loss, steps)
                    self.logger_run.log_metric("loss/targets", t_mean, steps)
                    self.logger_run.log_metric("loss/target_stds", t_std, steps)

            else:  # Fallback for other cases
                self.params, self.target_params, self.opt_state, loss = result[:4]
                if len(result) > 4 and self.prioritized_replay:
                    new_priorities = result[4]
                    self._update_priorities(data, new_priorities)

                if self.logger_run and (steps - self._last_log_step >= self.log_interval):
                    self._last_log_step = steps
                    self.logger_run.log_metric("loss/qloss", loss, steps)

        return loss

    def get_behavior_params(self):
        """Get parameters to use for behavior (training-time actions)."""
        return self.params

    def get_eval_params(self):
        """Get parameters to use for evaluation (eval-time actions)."""
        return self.get_behavior_params()

    def actions(self, obs, epsilon, eval_mode=False):
        # Select params: during eval with checkpointing prefer snapshot
        params_to_use = self.get_behavior_params()
        if eval_mode and self.use_checkpointing and self.checkpointing_enabled:
            params_to_use = self.checkpoint_params

        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    params_to_use, obs, next(self.key_seq) if self.param_noise else None
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"

        if self.param_noise:
            pass
        else:
            discription += f", epsilon : {self.update_eps:.3f}"

        if self.use_checkpointing and (self._last_ckpt_update_step is not None):
            discription += f", ckpt_upd_step : {int(self._last_ckpt_update_step)}"

        return discription

    def run_name_update(self, run_name):
        if self.munchausen:
            run_name = "M-" + run_name
        if (
            self.param_noise
            & self.dueling_model
            & self.double_q
            & self.n_step_method
            & self.prioritized_replay
        ):
            run_name = f"Rainbow({self.n_step} step)_" + run_name
        else:
            if self.param_noise:
                run_name = "Noisy_" + run_name
            if self.dueling_model:
                run_name = "Dueling_" + run_name
            if self.double_q:
                run_name = "Double_" + run_name
            if self.n_step_method:
                run_name = "{}Step_".format(self.n_step) + run_name
            if self.prioritized_replay:
                run_name = run_name + "+PER"
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="Q_network",
        run_name="Q_network",
    ):
        run_name = self.run_name_update(run_name)
        # Update log_interval to match the method parameter for consistency
        self.log_interval = log_interval

        if self.param_noise:
            self.exploration = ConstantSchedule(0)
        else:
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )
        self.update_eps = 1.0
        self.eval_freq = ((total_timesteps // 100) // self.worker_size) * self.worker_size

        pbar = trange(0, total_timesteps, self.worker_size, miniters=log_interval)
        self.logger = TensorboardLogger(run_name, experiment_name, self.log_dir, self)
        with self.logger as self.logger_run:
            if self.env_type == "SingleEnv":
                if self.use_checkpointing:
                    self.learn_SingleEnv_checkpointing(pbar, callback, log_interval)
                else:
                    self.learn_SingleEnv(pbar, callback, log_interval)
            if self.env_type == "VectorizedEnv":
                if self.use_checkpointing:
                    self.learn_VectorizedEnv_checkpointing(pbar, callback, log_interval)
                else:
                    self.learn_VectorizedEnv(pbar, callback, log_interval)

            self.eval(total_timesteps)

            self.save_params(self.logger_run.get_local_path("params"))

    def learn_SingleEnv(self, pbar, callback=None, log_interval=1000):
        obs, info = self.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.lossque = deque(maxlen=10)
        eval_result = None

        # Always run non-checkpointing flow; branching handled in learn()

        for steps in pbar:
            actions = self.actions(obs, self.update_eps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0][0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def learn_VectorizedEnv(self, pbar, callback=None, log_interval=1000):
        self.lossque = deque(maxlen=10)
        eval_result = None

        # Always run non-checkpointing flow; branching handled in learn()

        for steps in pbar:
            self.update_eps = self.exploration.value(steps)
            obs = self.env.current_obs()
            actions = self.actions([obs], self.update_eps)
            self.env.step(actions)

            if steps > self.learning_starts and steps % self.train_freq == 0:
                for idx in range(self.worker_size):
                    loss = self.train_step(steps + idx, self.gradient_steps)
                    self.lossque.append(loss)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            self.replay_buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def eval(self, steps):
        # Deterministic greedy evaluation using public actions() API.
        def eval_action_fn(obs):
            a = self.actions(obs, 0.0, eval_mode=True)
            arr = np.asarray(a)
            if arr.size == 1:
                return int(arr.item())
            else:
                return int(arr[0][0])

        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            eval_action_fn,
            logger_run=self.logger_run,
            steps=steps,
        )

    def learn_SingleEnv_checkpointing(self, pbar, callback=None, log_interval=1000, obs=None):
        # Initialize required state if not provided
        if obs is None:
            obs, info = self.env.reset()
            obs = [np.expand_dims(obs, axis=0)]
        self.lossque = deque(maxlen=10)
        eval_result = None

        score = 0.0
        eplen = 0

        def _ckpt_train_and_reset(step_val, accumulated_timesteps):
            # Exact training-parity via residual accumulation
            self._ckpt_update_residual += int(accumulated_timesteps)
            num_update_iters = 0
            while self._ckpt_update_residual >= self.train_freq:
                self._ckpt_update_residual -= self.train_freq
                num_update_iters += 1
            if num_update_iters > 0:
                total_updates = num_update_iters * self.gradient_steps
                loss_local = self.train_step(step_val, total_updates)
                self.lossque.append(loss_local)

        for steps in pbar:
            eplen += 1
            actions = self.actions(obs, self.update_eps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0][0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            score += float(reward)
            obs = next_obs

            if terminated or truncated:
                if steps > self.learning_starts:
                    ckpt_success = self._checkpoint_on_episode_end(
                        steps, score, eplen, train_and_reset_callback=_ckpt_train_and_reset
                    )
                else:
                    ckpt_success = True  # No checkpointing yet, so consider it successful
                score = 0.0
                eplen = 0

                # Use true_reset if checkpointing failed and true_reset is available
                if not ckpt_success and self._has_true_reset():
                    obs, info = self.env.true_reset()
                else:
                    obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            # Maintain epsilon schedule - only update on training steps for consistency with non-checkpointing flow
            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def learn_VectorizedEnv_checkpointing(self, pbar, callback=None, log_interval=1000):
        self.lossque = deque(maxlen=10)
        eval_result = None

        scores = np.zeros([self.worker_size], dtype=np.float64)
        eplens = np.zeros([self.worker_size], dtype=np.int32)

        def _ckpt_train_and_reset(step_val, accumulated_timesteps):
            # Exact training-parity via residual accumulation
            self._ckpt_update_residual += int(accumulated_timesteps)
            num_update_iters = 0
            while self._ckpt_update_residual >= self.train_freq:
                self._ckpt_update_residual -= self.train_freq
                num_update_iters += 1
            if num_update_iters > 0:
                total_updates = num_update_iters * self.gradient_steps
                loss_local = self.train_step(step_val, total_updates)
                self.lossque.append(loss_local)

        for steps in pbar:
            self.update_eps = self.exploration.value(steps)
            obs = self.env.current_obs()
            actions = self.actions([obs], self.update_eps)
            self.env.step(actions)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            scores += rewards
            eplens += 1

            self.replay_buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            if steps > self.learning_starts:
                done = np.logical_or(terminateds, truncateds)
                done_idx = np.where(done)[0]
                ckpt_results = []
                for idx in done_idx:
                    ckpt_success = self._checkpoint_on_episode_end(
                        steps, float(scores[idx]), int(eplens[idx]), _ckpt_train_and_reset
                    )
                    ckpt_results.append((idx, ckpt_success))
                    scores[idx] = 0.0
                    eplens[idx] = 0

                # Check if any checkpointing failed and true_reset is available
                # For vectorized environments, we apply true_reset to the entire environment
                # if any worker had a checkpointing failure
                if self._has_true_reset() and any(not success for _, success in ckpt_results):
                    # Reset the entire vectorized environment with true_reset
                    # This will affect all workers, but ensures consistency
                    try:
                        self.env.true_reset()
                    except Exception:
                        # Fallback to regular reset if true_reset fails
                        self.env.reset()

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def test(self, episode=10):
        with self.logger as self.logger_run:
            self.test_eval_env(episode)

    def test_eval_env(self, episode):
        # record_and_test expects (env_builder, logger_run, actions_eval_fn, episode, conv_action=None)
        actions_fn = (
            self.test_action
            if hasattr(self, "test_action")
            else (lambda obs: self.actions(obs, 0.0))
        )
        return record_and_test(
            self.env_builder,
            self.logger_run,
            actions_fn,
            episode,
            conv_action=None,
        )

    # -------------------------------
    # Checkpointing scaffolding hooks
    # -------------------------------
    def _maybe_enable_checkpointing(self, steps):
        if (
            self.use_checkpointing
            and (not self.checkpointing_enabled)
            and steps > self.steps_before_checkpointing
        ):
            # Relax the threshold slightly when entering checkpointing mode
            self._ckpt_max_eps_before_update = self.max_eps_before_checkpointing
            self.checkpointing_enabled = True

    def _checkpoint_update_snapshot(self):
        """Default checkpoint snapshot strategy for Q-Network family.

        This copies current network parameters into checkpoint snapshots.
        Subclasses can override this for custom snapshot strategies.
        """
        # Default strategy: snapshot eval parameters (mirrors eval behavior)
        if hasattr(self, "params"):
            self.checkpoint_params = deepcopy(self.get_eval_params())

    def _log_ckpt_snapshot_update(self, steps):
        """Record that a checkpoint snapshot was updated and log it."""
        self._last_ckpt_update_step = int(steps)
        self._ckpt_update_count += self.checkpointing_enabled
        if getattr(self, "logger_run", None) is not None:
            try:
                self.logger_run.log_metric(
                    "ckpt/ckpt_baseline", float(self._ckpt_baseline), int(steps)
                )
                self.logger_run.log_metric(
                    "ckpt/update_count", float(self._ckpt_update_count), int(steps)
                )
            except Exception:
                pass

    def _has_true_reset(self):
        """Check if the environment has true_reset method (from atari_wrappers)."""
        return hasattr(self.env, "true_reset") and callable(getattr(self.env, "true_reset", None))

    def _checkpoint_on_episode_end(
        self, steps, episode_return, episode_len, train_and_reset_callback=None
    ):
        """Generic per-episode checkpointing state update.

        Subclasses can call this at the end of an episode to drive a TD7-like
        checkpoint schedule. If a callback is provided, it will be called when
        it's time to perform a training/reset pulse.

        Returns:
            bool: True if checkpointing update was successful, False if failed (below baseline)
        """
        if not self.use_checkpointing:
            return True

        # Update runtime counters and statistics
        self._ckpt_eps_since_update += 1
        self._ckpt_timesteps_since_update += int(episode_len)
        self._ckpt_returns_window.append(float(episode_return))

        # Enable checkpointing mode once enough steps have elapsed
        self._maybe_enable_checkpointing(steps)

        # Compute robust window statistic (quantile on standardized returns if enabled)
        window_stat = compute_ckpt_window_stat(
            self._ckpt_returns_window,
            self.ckpt_baseline_q,
            self.use_ckpt_return_standardization,
            self.ckpt_baseline_mode,
        )
        if window_stat is None:
            return True
        self.logger_run.log_metric("ckpt/window_stat", float(window_stat), int(steps))

        # Pre-enable phase: warm-up baseline/snapshot, no training pulses by default
        if not self.checkpointing_enabled:
            self._checkpoint_update_snapshot()
            if callable(train_and_reset_callback):
                train_and_reset_callback(steps, self._ckpt_timesteps_since_update)
            self._ckpt_eps_since_update = 0
            self._ckpt_timesteps_since_update = 0
            self._ckpt_returns_window = []
            return True

        if window_stat < self._ckpt_baseline:
            # Checkpointing update failed - return False to trigger true_reset if available
            if callable(train_and_reset_callback):
                train_and_reset_callback(steps, self._ckpt_timesteps_since_update)
            self._ckpt_eps_since_update = 0
            self._ckpt_timesteps_since_update = 0
            self._ckpt_returns_window = []
            return False

        # Enabled phase: end-of-window refresh with training pulse
        if self._ckpt_eps_since_update >= self._ckpt_max_eps_before_update:
            self._checkpoint_update_snapshot()
            self._ckpt_baseline = window_stat
            self._log_ckpt_snapshot_update(steps)
            if callable(train_and_reset_callback):
                train_and_reset_callback(steps, self._ckpt_timesteps_since_update)
            self._ckpt_eps_since_update = 0
            self._ckpt_timesteps_since_update = 0
            self._ckpt_returns_window = []

        return True
