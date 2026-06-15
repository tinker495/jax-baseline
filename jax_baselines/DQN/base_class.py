from copy import deepcopy

import jax
import numpy as np

from jax_baselines.common.checkpoint import make_checkpoint_scaffold
from jax_baselines.common.env_info import get_local_env_info
from jax_baselines.common.eval import evaluate_policy, record_and_test
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.replay_protocol import (
    ReplayBufferFactory,
    require_replay_factory,
)
from jax_baselines.common.rollout import (
    ActionSelection,
    CheckpointTrainPulse,
    RolloutSpec,
)
from jax_baselines.common.rollout_stats import EpisodeTracker
from jax_baselines.common.schedules import ConstantSchedule, LinearSchedule
from jax_baselines.common.seeding import key_gen, set_global_seeds
from jax_baselines.common.serialization import restore, save
from jax_baselines.common.training_session import TrainingSession, off_policy_loop
from jax_baselines.DQN.training import QNetTrainingLifecycle


class Q_Network_Family(object):
    _qnet_handles_train_pulse = False

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
        replay_factory: ReplayBufferFactory | None = None,
        # Checkpointing options (opt-in by default for base class)
        use_checkpointing=True,
        steps_before_checkpointing=500000,
        max_eps_before_checkpointing=10,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="median",
        ckpt_baseline_q=None,
    ):
        self.name = "Q_Network_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        set_global_seeds(self.seed)
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
        self.replay_factory = replay_factory

        self.get_env_setup()
        self.get_memory_setup()

        # Generic checkpointing scaffolding (used by algorithms that opt-in)
        self.use_checkpointing = use_checkpointing
        self.steps_before_checkpointing = min(int(steps_before_checkpointing), learning_starts * 2)
        self.max_eps_before_checkpointing = int(max_eps_before_checkpointing)
        self.initial_checkpoint_window = int(initial_checkpoint_window)

        self.logger_run = None
        self.rollout_tracker = None
        self._ckpt_update_residual = 0
        self.ckpt = make_checkpoint_scaffold(
            use_checkpointing=self.use_checkpointing,
            steps_before_checkpointing=self.steps_before_checkpointing,
            max_eps_before_checkpointing=self.max_eps_before_checkpointing,
            initial_checkpoint_window=self.initial_checkpoint_window,
            ckpt_baseline_mode=ckpt_baseline_mode,
            ckpt_baseline_q=ckpt_baseline_q,
            snapshot=self._checkpoint_update_snapshot,
            log_metric=lambda key, value, step: (
                self.logger_run.log_metric(key, value, step)
                if self.logger_run is not None
                else None
            ),
        )

        # Logging throttle based on last log step
        self._last_log_step = 0
        self.training_lifecycle = QNetTrainingLifecycle(self)

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
        ) = get_local_env_info(self.env_builder, self.num_workers, seed=self.seed)
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")

    def get_memory_setup(self):
        replay_factory = require_replay_factory(self.replay_factory, "ReplayBufferFactory")
        self.replay_buffer = replay_factory(
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

    def train_step(self, steps, gradient_steps):
        return self.training_lifecycle.train(steps, gradient_steps)

    def _train_on_batch(self, data, context):
        """Run one algorithm-specific update and return a QNetTrainResult."""
        raise NotImplementedError

    def _aggregate_train_reports(self, reports):
        return reports[-1]

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

    def get_behavior_params(self):
        """Get parameters to use for behavior (training-time actions)."""
        return self.params

    def get_eval_params(self):
        """Get parameters to use for evaluation (eval-time actions)."""
        return self.get_behavior_params()

    def actions(self, obs, epsilon, eval_mode=False):
        # Select params: during eval with checkpointing prefer snapshot
        params_to_use = self.get_behavior_params()
        if eval_mode and self.use_checkpointing and self.ckpt.enabled:
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

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(self.lossque):.3f}"

        if not self.param_noise:
            description += f", epsilon : {self.update_eps:.3f}"

        if self.use_checkpointing and (self.ckpt.last_update_step is not None):
            description += f", ckpt_upd_step : {int(self.ckpt.last_update_step)}"

        description += self._rollout_pbar_suffix()

        return description

    def _rollout_pbar_suffix(self):
        """Pbar fragment with the rollout window-mean reward (empty until the
        first training episode completes)."""
        if self.rollout_tracker is None:
            return ""
        fragment = self.rollout_tracker.describe()
        return f", {fragment}" if fragment else ""

    def run_name_update(self, run_name):
        if self.munchausen:
            run_name = "M-" + run_name
        if (
            self.param_noise
            and self.dueling_model
            and self.double_q
            and self.n_step_method
            and self.prioritized_replay
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
        eval_num=100,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    ):
        return TrainingSession().run(
            self,
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
            eval_num,
            logger_factory=logger_factory,
            progress_factory=progress_factory,
            record_test_fn=record_test_fn,
        )

    def prepare_run(self, total_timesteps):
        if self.param_noise:
            self.exploration = ConstantSchedule(0)
        else:
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )
        self.update_eps = 1.0

    def run_training_loop(self, ctx):
        off_policy_loop(self, ctx)

    # -------------------------------
    # Rollout seam (RolloutSpec wiring)
    # -------------------------------
    def _bind_loss_window(self, window):
        self.lossque = window

    def _single_action_selection(self, obs, steps):
        actions = self.actions(obs, self.update_eps)
        return ActionSelection(env_action=actions[0][0], store_action=actions[0])

    def _vector_action_selection(self, obs, steps):
        actions = self.actions([obs], self.update_eps)
        return ActionSelection(env_action=actions, store_action=actions)

    def _refresh_exploration(self, steps):
        self.update_eps = self.exploration.value(steps)

    def make_rollout_spec(self, ctx):
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        pulse = CheckpointTrainPulse(
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            train=self.train_step,
            record_loss=lambda loss: self.lossque.append(loss),
            read_residual=lambda: self._ckpt_update_residual,
            write_residual=lambda value: setattr(self, "_ckpt_update_residual", value),
        )
        return RolloutSpec(
            env=self.env,
            replay_buffer=self.replay_buffer,
            learning_starts=self.learning_starts,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            eval_freq=ctx.eval_freq,
            worker_size=self.worker_size,
            single_action=self._single_action_selection,
            vector_action=self._vector_action_selection,
            refresh_exploration=self._refresh_exploration,
            has_true_reset=self._has_true_reset,
            train=self.train_step,
            evaluate=lambda steps: self.eval(ctx, steps),
            describe=self.description,
            bind_loss_window=self._bind_loss_window,
            record_rollout_episode=self.rollout_tracker.record,
            checkpoint_on_episode_end=self.ckpt.on_episode_end,
            checkpoint_pulse=pulse,
        )

    def eval(self, ctx, steps):
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
            logger_run=ctx.logger_run,
            steps=steps,
        )

    def test(self, episode=10):
        with self.logger as self.logger_run:
            self.test_eval_env(episode)

    def test_eval_env(self, episode):
        # record_and_test expects (env_builder, logger_run, actions_eval_fn, episode, conv_action=None)
        record_test_fn = getattr(self, "record_test_fn", record_and_test)
        return record_test_fn(
            self.env_builder,
            self.logger_run,
            lambda obs: self.actions(obs, 0.0),
            episode,
            conv_action=None,
        )

    # -------------------------------
    # Checkpointing scaffolding hooks
    # -------------------------------
    def _checkpoint_update_snapshot(self):
        """Snapshot eval parameters into `checkpoint_params`.

        Subclasses override to customise which parameters are snapshotted.
        """
        self.checkpoint_params = deepcopy(self.get_eval_params())

    def _has_true_reset(self):
        return False
