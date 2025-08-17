from collections import deque
from copy import deepcopy

import numpy as np
from tqdm.auto import trange

from jax_baselines.common.env_info import get_local_env_info
from jax_baselines.common.eval import evaluate_policy, record_and_test
from jax_baselines.common.logger import TensorboardLogger
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.replay_factory import make_replay_buffer
from jax_baselines.common.utils import RunningMeanStd, key_gen, restore, save


class Deteministic_Policy_Gradient_Family(object):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        scaled_by_reset=False,
        simba=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer="adamw",
        # Checkpointing options (opt-in by default for base class)
        use_checkpointing=True,
        steps_before_checkpointing=500000,
        max_eps_before_checkpointing=5,
        initial_checkpoint_window=1,
    ):
        self.name = "Deteministic_Policy_Gradient_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.train_steps_count = 0
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_tau = target_network_update_tau
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._gamma = self.gamma**n_step  # n_step gamma
        self.log_dir = log_dir
        self.n_step_method = n_step > 1
        self.n_step = n_step
        self.scaled_by_reset = scaled_by_reset
        self.reset_freq = 500000
        self.simba = simba
        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-3 / self.batch_size)

        self.get_env_setup()
        self.get_memory_setup()

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()

        if self.simba:
            self.obs_rms = RunningMeanStd(shapes=self.observation_space, dtype=np.float64)

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
        self._ckpt_baseline = None  # window-stat baseline (initialized on first window)
        self._ckpt_update_residual = 0  # exact training-parity residual accumulator
        # Track snapshot updates for logging and progress description
        self._last_ckpt_update_step = None
        self._ckpt_update_count = 0

        # Logging throttle based on last log step
        self._last_log_step = 0

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        # Use common helper to standardize environment info
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
        # Use common replay factory to pick the correct buffer type
        self.replay_buffer = make_replay_buffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_shape_or_n=self.action_size,
            worker_size=self.worker_size,
            n_step=self.n_step if self.n_step_method else 1,
            gamma=self.gamma,
            prioritized=self.prioritized_replay,
            alpha=self.prioritized_replay_alpha,
            eps=self.prioritized_replay_eps,
        )

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def actions(self, obs, steps, eval=False):
        pass

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"
        if self.use_checkpointing and (self._last_ckpt_update_step is not None):
            discription += f", ckpt_upd_step : {int(self._last_ckpt_update_step)}"
        return discription

    def run_name_update(self, run_name):
        if self.simba:
            run_name = "Simba_" + run_name
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
        experiment_name="DPG_network",
        run_name="DPG_network",
    ):
        run_name = self.run_name_update(run_name)
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
            actions = self.actions(obs, steps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.learning_starts and steps % self.train_freq == 0:
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
            obs = self.env.current_obs()
            actions = self.actions([obs], steps)
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
        # Evaluation should use the public actions() API with eval=True so that
        # subclasses can implement snapshot-aware behavior consistently.
        def eval_action_fn(obs):
            return self.actions(obs, steps, eval=True)

        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            eval_action_fn,
            logger_run=self.logger_run,
            steps=steps,
        )

    def test(self, episode=10, run_name=None):
        with self.logger as self.logger_run:
            self.test_eval_env(episode)

    def test_action(self, obs):
        return self.actions(obs, np.inf)

    def test_eval_env(self, episode):
        # Use common test helper: (env_builder, logger_run, actions_eval_fn, episode, conv_action=None)
        return record_and_test(
            self.env_builder,
            self.logger_run,
            self.test_action,
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
        """Default checkpoint snapshot strategy for DPG family.

        This copies current policy and encoder parameters into checkpoint snapshots.
        Subclasses can override this for custom snapshot strategies.
        """
        # Default strategy: snapshot policy and encoder params if they exist
        if hasattr(self, "policy_params"):
            self.checkpoint_policy_params = deepcopy(self.policy_params)
        if hasattr(self, "fixed_encoder_params"):
            self.checkpoint_encoder_params = deepcopy(self.fixed_encoder_params)
        elif hasattr(self, "encoder_params"):
            self.checkpoint_encoder_params = deepcopy(self.encoder_params)

    def _log_ckpt_snapshot_update(self, steps):
        """Record that a checkpoint snapshot was updated and log it."""
        self._last_ckpt_update_step = int(steps)
        self._ckpt_update_count += 1
        if getattr(self, "logger_run", None) is not None:
            try:
                self.logger_run.log_metric("ckpt/snapshot_update", 1.0, int(steps))
                self.logger_run.log_metric(
                    "ckpt/update_count", float(self._ckpt_update_count), int(steps)
                )
            except Exception:
                pass

    def _checkpoint_on_episode_end(
        self, steps, episode_return, episode_len, train_and_reset_callback=None
    ):
        """Generic per-episode checkpointing state update.

        Subclasses can call this at the end of an episode to drive a TD7-like
        checkpoint schedule. If a callback is provided, it will be called when
        it's time to perform a training/reset pulse.
        """
        if not self.use_checkpointing:
            return

        # Update runtime counters and statistics
        self._ckpt_eps_since_update += 1
        self._ckpt_timesteps_since_update += int(episode_len)
        self._ckpt_returns_window.append(float(episode_return))

        # Enable checkpointing mode once enough steps have elapsed
        self._maybe_enable_checkpointing(steps)

        # Compute robust window statistic (quantile on standardized returns if enabled)
        window_stat = self._compute_ckpt_window_stat()
        if window_stat is None:
            return

        # Initialize baseline once we obtain the first valid window statistic
        if self._ckpt_baseline is None:
            self._ckpt_baseline = window_stat

        # Pre-enable phase: warm-up baseline/snapshot, no training pulses by default
        if not self.checkpointing_enabled:
            # Initialize baseline once we obtain the first valid window statistic
            if self._ckpt_baseline is None:
                self._ckpt_baseline = window_stat
            self._checkpoint_update_snapshot()
            self._log_ckpt_snapshot_update(steps)
            if callable(train_and_reset_callback):
                train_and_reset_callback(steps, self._ckpt_timesteps_since_update)
            self._ckpt_eps_since_update = 0
            self._ckpt_timesteps_since_update = 0
            self._ckpt_returns_window = []

        if (self._ckpt_baseline is not None) and (window_stat < self._ckpt_baseline):
            if callable(train_and_reset_callback):
                train_and_reset_callback(steps, self._ckpt_timesteps_since_update)
            self._ckpt_eps_since_update = 0
            self._ckpt_timesteps_since_update = 0
            self._ckpt_returns_window = []
            return

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

    def _compute_ckpt_window_stat(self):
        """Compute robust window statistic for checkpoint decisions.

        Uses q-quantile of returns in the current window. If
        use_ckpt_return_standardization is True, standardizes using
        median and MAD within the window before quantile.
        """
        if not self._ckpt_returns_window:
            return None
        arr = np.asarray(self._ckpt_returns_window, dtype=np.float64)
        if self.use_ckpt_return_standardization and arr.size >= 3:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            scale = mad if mad > 1e-8 else 1.0
            arr = (arr - med) / scale
        q = float(self.ckpt_quantile)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(arr, q))

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
            actions = self.actions(obs, steps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            score += float(reward)
            obs = next_obs

            if terminated or truncated:
                if steps > self.learning_starts:
                    self._checkpoint_on_episode_end(
                        steps, score, eplen, train_and_reset_callback=_ckpt_train_and_reset
                    )
                score = 0.0
                eplen = 0
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

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
            obs = self.env.current_obs()
            actions = self.actions([obs], steps)
            self.env.step(actions)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            # accumulate
            scores += rewards
            eplens += 1

            self.replay_buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            if steps > self.learning_starts:
                done_mask = np.logical_or(terminateds, truncateds)
                done_indices = np.where(done_mask)[0]
                for idx in done_indices:
                    self._checkpoint_on_episode_end(
                        steps, float(scores[idx]), int(eplens[idx]), _ckpt_train_and_reset
                    )
                    scores[idx] = 0.0
                    eplens[idx] = 0

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))
