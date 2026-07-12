import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.checkpoint import make_checkpoint_scaffold, snapshot_pytree
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.env_info import get_local_env_info
from jax_baselines.core.eval import evaluate_policy, record_and_test
from jax_baselines.core.replay_protocol import (
    LocalReplayNeed,
    PriorityNeed,
    ReplayBufferFactory,
    require_replay_factory,
)
from jax_baselines.core.rollout import (
    ActionSelection,
    CheckpointTrainPulse,
    RolloutSpec,
)
from jax_baselines.core.rollout_stats import EpisodeTracker
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.core.training_session import TrainingSession, off_policy_loop
from jax_baselines.DQN.training import (
    QNetTrainingLifecycle,
    QNetTrainReport,
    QNetTrainResult,
)
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class Q_Network_Family:
    _run_name = "Q_network"

    supports_bulk_training = False

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
        max_bulk_updates_per_pulse=32,
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
        optimizer_factory: OptimizerFactory | None = None,
        compress_memory=False,
        replay_factory: ReplayBufferFactory | None = None,
        # Checkpointing options (opt-in by default for base class)
        use_checkpointing=True,
        steps_before_checkpointing=500000,
        max_eps_before_checkpointing=10,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="median",
        ckpt_baseline_q=None,
        checkpoint_store: CheckpointStore | None = None,
    ):
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
        self.max_bulk_updates_per_pulse = max_bulk_updates_per_pulse
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
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.optimizer = self._make_optimizer(self.learning_rate)

        self.compress_memory = compress_memory
        self.replay_factory = replay_factory
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)

        self.get_env_setup()
        self.get_memory_setup()

        # Generic checkpointing scaffolding (used by algorithms that opt-in)
        self.use_checkpointing = use_checkpointing
        self.steps_before_checkpointing = min(int(steps_before_checkpointing), learning_starts * 2)
        self.max_eps_before_checkpointing = int(max_eps_before_checkpointing)
        self.initial_checkpoint_window = int(initial_checkpoint_window)

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
        )
        self.baseline_mode = self.ckpt.baseline_mode
        self.baseline_q = self.ckpt.baseline_q

        # Logging throttle based on last log step
        self._last_log_step = 0
        self.training_lifecycle = QNetTrainingLifecycle(self)

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            # Calls overridden setup_model in children
            self.setup_model()

    def save_params(self, path):
        self.checkpoint_store.save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = self.checkpoint_store.restore(path)

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
        priority = (
            PriorityNeed(alpha=self.prioritized_replay_alpha, eps=self.prioritized_replay_eps)
            if self.prioritized_replay
            else None
        )
        self.replay_buffer = replay_factory(
            LocalReplayNeed(
                buffer_size=self.buffer_size,
                observation_space=self.observation_space,
                action_shape_or_n=1,
                worker_size=self.worker_size,
                n_step=self.n_step if self.n_step_method else 1,
                gamma=self.gamma,
                priority=priority,
                compress_observations=self.compress_memory,
            )
        )

    def setup_model(self):
        pass

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

    def train_step(self, steps, gradient_steps, logger_run=None, log_interval=None):
        return self.training_lifecycle.train(steps, gradient_steps, logger_run, log_interval)

    def _train_on_batch(self, data, context):
        (
            self.params,
            self.target_params,
            self.opt_state,
            loss,
            target,
            priorities,
        ) = self._train_step(
            self.params,
            self.target_params,
            self.opt_state,
            context.train_steps_count,
            next(self.key_seq) if self.param_noise else None,
            **data,
        )
        return QNetTrainResult.from_values(loss=loss, target=target, replay_priorities=priorities)

    def _train_on_bulk(self, data, contexts):
        steps = jnp.asarray([context.train_steps_count for context in contexts])
        keys = jax.random.split(next(self.key_seq), len(contexts)) if self.param_noise else None
        carry = (self.params, self.target_params, self.opt_state)
        (
            (self.params, self.target_params, self.opt_state),
            (
                losses,
                targets,
                priorities,
            ),
        ) = self._bulk_scan(carry, keys, steps, data)
        return QNetTrainResult.from_values(
            loss=jnp.mean(losses),
            target=jnp.mean(targets),
            replay_priorities=priorities,
            update_count=len(contexts),
        )

    def _bulk_scan(self, carry, keys, steps, data):
        def train_one(carry, xs):
            params, target_params, opt_state = carry
            if self.param_noise:
                step, key, batch = xs
            else:
                step, batch = xs
                key = None
            params, target_params, opt_state, loss, target, priorities = self._train_step(
                params,
                target_params,
                opt_state,
                step,
                key,
                **batch,
            )
            return (params, target_params, opt_state), (loss, target, priorities)

        xs = (steps, keys, data) if self.param_noise else (steps, data)
        return jax.lax.scan(train_one, carry, xs)

    def _aggregate_train_reports(self, reports):
        if len(reports) == 1:
            return reports[-1]
        counts = jnp.array([report.update_count for report in reports])
        total = jnp.sum(counts)
        metrics = {
            name: jnp.sum(jnp.array([report.metrics[name] for report in reports]) * counts) / total
            for name in reports[-1].metrics
            if all(name in report.metrics for report in reports)
        }
        histograms = {
            name: jnp.sum(
                jnp.stack([report.histograms[name] for report in reports])
                * counts.reshape((-1, *([1] * len(reports[-1].histograms[name].shape)))),
                axis=0,
            )
            / total
            for name in reports[-1].histograms
            if all(name in report.histograms for report in reports)
        }
        target = None
        if all(report.target is not None for report in reports):
            target = jnp.sum(jnp.array([report.target for report in reports]) * counts) / total
        return QNetTrainReport(
            loss=jnp.sum(jnp.array([report.loss for report in reports]) * counts) / total,
            target=target,
            metrics=metrics,
            histograms=histograms,
            update_count=int(total),
        )

    def _get_actions(self, params, obses) -> np.ndarray:
        raise NotImplementedError

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

    def _random_actions(self, shape=None):
        if shape is None:
            shape = (self.worker_size, 1)
        return np.random.choice(self.action_size[0], shape)

    def _epsilon_greedy_actions(self, greedy_actions, epsilon):
        greedy_actions = np.asarray(greedy_actions)
        if epsilon <= 0:
            return greedy_actions
        random_actions = self._random_actions(greedy_actions.shape)
        if epsilon >= 1:
            return random_actions
        random_mask = np.random.uniform(size=greedy_actions.shape) < epsilon
        return np.where(random_mask, random_actions, greedy_actions)

    def actions(self, obs, epsilon, eval_mode=False):
        # Select params: during eval with checkpointing prefer snapshot
        params_to_use = self.get_behavior_params()
        if eval_mode and self.use_checkpointing and self.ckpt.enabled:
            params_to_use = self.checkpoint_params

        if epsilon >= 1:
            return self._random_actions()

        greedy_actions = self._get_actions(
            params_to_use, obs, next(self.key_seq) if self.param_noise else None
        )
        return self._epsilon_greedy_actions(greedy_actions, epsilon)

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
        experiment_name=None,
        run_name=None,
        eval_num=100,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    ):
        if experiment_name is None:
            experiment_name = self._run_name
        if run_name is None:
            run_name = self._run_name
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
            self.exploration = optax.constant_schedule(0)
        else:
            self.exploration = optax.linear_schedule(
                init_value=self.exploration_initial_eps,
                end_value=self.exploration_final_eps,
                transition_steps=int(self.exploration_fraction * total_timesteps),
            )
        self.update_eps = 1.0

    def run_training_loop(self, ctx):
        off_policy_loop(self, ctx)

    def release_run_context(self):
        self.rollout_tracker = None

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
        self.update_eps = float(self.exploration(steps))

    def make_rollout_spec(self, ctx):
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        train = lambda steps, gradient_steps: self.train_step(  # noqa: E731
            steps, gradient_steps, ctx.logger_run, ctx.log_interval
        )
        pulse = CheckpointTrainPulse(
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            train=train,
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
            train=train,
            evaluate=lambda steps: self.eval(ctx, steps),
            describe=self.description,
            bind_loss_window=self._bind_loss_window,
            record_rollout_episode=self.rollout_tracker.record,
            checkpoint_on_episode_end=lambda *args, **kwargs: self.ckpt.on_episode_end(
                *args, log_metric=ctx.logger_run.log_metric, **kwargs
            ),
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
        with self.logger as logger_run:
            self.test_eval_env(logger_run, episode)

    def test_eval_env(self, logger_run, episode):
        # record_and_test expects (env_builder, logger_run, actions_eval_fn, episode, conv_action=None)
        record_test_fn = getattr(self, "record_test_fn", record_and_test)
        return record_test_fn(
            self.env_builder,
            logger_run,
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
        self.checkpoint_params = snapshot_pytree(self.get_eval_params())

    def _has_true_reset(self):
        return False
