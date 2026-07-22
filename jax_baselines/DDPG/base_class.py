from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.checkpoint import make_checkpoint_scaffold, snapshot_pytree
from jax_baselines.core.checkpoint_state import CheckpointState
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
from jax_baselines.DDPG.training import DPGTrainingLifecycle, DPGTrainReport
from jax_baselines.math.statistics import RewardNormalizer, RunningMeanStd
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class Deteministic_Policy_Gradient_Family(object):
    _run_name = "DPG_network"

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
        train_freq=1,
        gradient_steps=1,
        max_bulk_updates_per_pulse=32,
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
        simba_v2=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory: OptimizerFactory | None = None,
        replay_factory: ReplayBufferFactory | None = None,
        # Checkpointing options (opt-in by default for base class)
        use_checkpointing=True,
        steps_before_checkpointing=500000,
        max_eps_before_checkpointing=20,
        initial_checkpoint_window=1,
        ckpt_baseline_mode="min",
        ckpt_baseline_q=None,
        checkpoint_store: CheckpointStore | None = None,
        reward_normalization=False,
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

        self.train_steps_count = 0
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.max_bulk_updates_per_pulse = max_bulk_updates_per_pulse
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
        self.simba = simba or simba_v2
        self.simba_v2 = simba_v2
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.optimizer = self._make_optimizer(self.learning_rate)
        self.replay_factory = replay_factory
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)
        self.reward_normalization = bool(reward_normalization)

        self.get_env_setup()
        self.reward_normalizer = (
            RewardNormalizer(self.worker_size, self.gamma) if self.reward_normalization else None
        )
        self.get_memory_setup()

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()

        self.eval_snapshot = None
        if self.simba:
            self.obs_rms = RunningMeanStd(shapes=self.observation_space, dtype=np.float64)
            self.action_obs_rms = None
            self.checkpoint_obs_rms = None

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
        self.training_lifecycle = DPGTrainingLifecycle(self)

    def save_params(self, path):
        self.checkpoint_store.save(path, self._build_checkpoint_state())

    def load_params(self, path):
        """Warm-start saved model, schedule, and normalization state.

        Optimizer, PRNG, replay, and environment state start fresh; this method
        does not reproduce an interrupted execution exactly.
        """
        self._restore_checkpoint_state(self.checkpoint_store.restore(path))

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

    # -------------------------------
    # Checkpoint contract (per-algorithm bundle = the real seam)
    # -------------------------------
    def checkpoint_params(self):
        """Return this algorithm's typed checkpoint param bundle (a flax.struct).

        Each concrete algorithm owns the bundle type and the fields it carries;
        the base composes the family-wide spine around it without naming them.
        """
        raise NotImplementedError

    def load_checkpoint_params(self, bundle):
        """Restore this algorithm's network params from its bundle."""
        raise NotImplementedError

    def _build_checkpoint_state(self) -> CheckpointState:
        return CheckpointState(
            params=self.checkpoint_params(),
            train_steps_count=np.asarray(self.train_steps_count, dtype=np.int64),
            ckpt_residual=np.asarray(self._ckpt_update_residual, dtype=np.float32),
            controller_state=self.ckpt.to_state(),
            eval_snapshot=self.eval_snapshot,
            obs_rms_state=self.obs_rms.to_state() if self.simba else None,
            action_obs_rms_state=(
                self.action_obs_rms.to_state()
                if (self.simba and self.action_obs_rms is not None)
                else None
            ),
            checkpoint_obs_rms_state=(
                self.checkpoint_obs_rms.to_state()
                if (self.simba and self.checkpoint_obs_rms is not None)
                else None
            ),
            reward_rms_state=(
                self.reward_normalizer.to_state() if self.reward_normalizer is not None else None
            ),
        )

    def _restore_checkpoint_state(self, state: CheckpointState):
        self.load_checkpoint_params(state.params)
        self.train_steps_count = int(np.asarray(state.train_steps_count).item())
        self._ckpt_update_residual = float(np.asarray(state.ckpt_residual).item())
        self.ckpt.from_state(state.controller_state)
        self.eval_snapshot = state.eval_snapshot

        if self.simba:
            if state.obs_rms_state is not None:
                self.obs_rms = RunningMeanStd.from_state(state.obs_rms_state)
            self.action_obs_rms = (
                RunningMeanStd.from_state(state.action_obs_rms_state)
                if state.action_obs_rms_state is not None
                else None
            )
            self.checkpoint_obs_rms = (
                RunningMeanStd.from_state(state.checkpoint_obs_rms_state)
                if state.checkpoint_obs_rms_state is not None
                else None
            )

        # getattr guards against pre-reward_rms checkpoints, not against self.
        reward_rms_state = getattr(state, "reward_rms_state", None)
        if self.reward_normalizer is not None:
            self.reward_normalizer.reset()
            if reward_rms_state is not None:
                self.reward_normalizer.restore(reward_rms_state)

    def get_env_setup(self):
        # Use common helper to standardize environment info
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
                action_shape_or_n=self.action_size,
                worker_size=self.worker_size,
                n_step=self.n_step if self.n_step_method else 1,
                gamma=self.gamma,
                priority=priority,
            )
        )

    def setup_model(self):
        pass

    def _setup_entropy_coef(self):
        """Initialize log_ent_coef and auto_entropy from self._ent_coef.

        Shared by the SAC-style stochastic actors (SAC, TQC, CrossQ). An
        "auto" coef enables automatic temperature tuning; an "auto_<x>" suffix
        seeds the initial value, and a numeric coef pins a fixed temperature.
        """
        if isinstance(self._ent_coef, str) and self._ent_coef.startswith("auto"):
            init_value = np.log(1e-1)
            if "_" in self._ent_coef:
                initial_alpha = float(self._ent_coef.split("_")[1])
                assert initial_alpha > 0.0, "The initial value of ent_coef must be greater than 0"
                init_value = np.log(initial_alpha)
            self.log_ent_coef = jax.device_put(init_value)
            self.auto_entropy = True
        else:
            try:
                self.log_ent_coef = jnp.log(float(self._ent_coef))
            except ValueError as err:
                raise ValueError(f"Invalid value for ent_coef: {self._ent_coef}") from err
            self.auto_entropy = False

        self.ent_coef_optimizer = optax.adam(self.ent_coef_learning_rate)
        self.opt_ent_coef_state = self.ent_coef_optimizer.init(self.log_ent_coef)

    def _train_ent_coef(self, log_coef, opt_state, log_prob):
        def loss(log_ent_coef):
            entropy = -jax.lax.stop_gradient(log_prob)
            return jnp.mean(jnp.exp(log_ent_coef) * (entropy - self.target_entropy))

        grad = jax.grad(loss)(log_coef)
        updates, opt_state = self.ent_coef_optimizer.update(grad, opt_state, log_coef)
        return optax.apply_updates(log_coef, updates), opt_state

    def train_step(self, steps, gradient_steps, logger_run=None, log_interval=None):
        return self.training_lifecycle.train(steps, gradient_steps, logger_run, log_interval)

    def prepare_run(self, total_timesteps):
        pass

    def run_training_loop(self, ctx):
        off_policy_loop(self, ctx)

    def _train_on_batch(self, data, context):
        raise NotImplementedError

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
        target = None
        if all(report.target is not None for report in reports):
            target = jnp.sum(jnp.array([report.target for report in reports]) * counts) / total
        return DPGTrainReport(
            loss=jnp.sum(jnp.array([report.loss for report in reports]) * counts) / total,
            target=target,
            metrics=metrics,
            update_count=int(total),
        )

    def get_behavior_state(self):
        """Get state dict to use for behavior (training-time actions).

        The DPG family acts from a deterministic/stochastic policy network only;
        the encoder slot is unused here and TD7 overrides to fill it.
        """
        return {
            "encoder": None,
            "policy": self.policy_params,
        }

    def get_eval_state(self):
        """Get state dict to use for evaluation (eval-time actions)."""
        return self.get_behavior_state()

    def actions(self, obs, steps, eval=False):
        obs = self._apply_simba_normalization(obs, eval, steps)
        if not eval and steps <= self.learning_starts:
            return self._random_warmup_actions(eval=eval)
        state = self._select_action_state(eval, steps)
        actions = self._policy_action_from_state(state, obs, eval, steps)
        return self._apply_action_noise(actions, steps, eval)

    def _random_warmup_actions(self, eval=False):
        worker_size = 1 if eval else self.worker_size
        return np.random.uniform(-1.0, 1.0, size=(worker_size, self.action_size[0]))

    def _select_action_state(self, eval, steps):
        if eval and self.use_checkpointing and self.ckpt.enabled and self.eval_snapshot is not None:
            return self.eval_snapshot
        return self.get_behavior_state()

    def _policy_action_from_state(self, state, obs, eval, steps):
        if eval:
            return np.asarray(self._get_eval_actions(state["policy"], obs))
        return np.asarray(self._get_actions(state["policy"], obs, next(self.key_seq)))

    def _apply_action_noise(self, actions, steps, eval):
        return actions

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(self.lossque):.3f}"
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
        if self.simba_v2:
            run_name = "SimbaV2_" + run_name
        elif self.simba:
            run_name = "Simba_" + run_name
        if self.n_step_method:
            run_name = "{}Step_".format(self.n_step) + run_name
        if self.prioritized_replay:
            run_name = run_name + "+PER"
        return run_name

    def _apply_simba_normalization(self, obs, eval, steps):
        if self.simba:
            rms = (
                self.checkpoint_obs_rms
                if (
                    eval
                    and self.use_checkpointing
                    and self.ckpt.enabled
                    and self.checkpoint_obs_rms is not None
                )
                else self._policy_update_obs_rms()
            )
            if (not eval) and steps != np.inf:
                self.obs_rms.update(obs)
            obs = rms.normalize(obs)
        return obs

    def _policy_update_obs_rms(self):
        return self.action_obs_rms if self.action_obs_rms is not None else self.obs_rms

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

    # -------------------------------
    # Rollout seam (RolloutSpec wiring)
    # -------------------------------
    def _bind_loss_window(self, window):
        self.lossque = window

    def _single_action_selection(self, obs, steps):
        actions = self.actions(obs, steps)
        return ActionSelection(env_action=actions[0], store_action=actions[0])

    def _vector_action_selection(self, obs, steps):
        actions = self.actions(obs, steps)
        return ActionSelection(env_action=actions, store_action=actions)

    def _snapshot_action_normalizer(self):
        if self.simba:
            self.action_obs_rms = deepcopy(self.obs_rms)

    def _write_ckpt_residual(self, value):
        self._ckpt_update_residual = value

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
            write_residual=self._write_ckpt_residual,
            post_pulse=self._snapshot_action_normalizer,
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
            refresh_exploration=lambda steps: None,
            force_reset=None,
            train=train,
            evaluate=lambda steps: self.eval(ctx, steps),
            describe=self.description,
            bind_loss_window=self._bind_loss_window,
            record_rollout_episode=self.rollout_tracker.record,
            checkpoint_on_episode_end=lambda *args, **kwargs: self.ckpt.on_episode_end(
                *args, log_metric=ctx.logger_run.log_metric, **kwargs
            ),
            checkpoint_pulse=pulse,
            reward_normalization=self.reward_normalization,
            record_transition=(
                self.reward_normalizer.record if self.reward_normalizer is not None else None
            ),
        )

    def eval(self, ctx, steps):
        # Evaluation should use the public actions() API with eval=True so that
        # subclasses can implement snapshot-aware behavior consistently.
        def eval_action_fn(obs):
            return self.actions(obs, steps, eval=True)

        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            eval_action_fn,
            logger_run=ctx.logger_run,
            steps=steps,
        )

    def test(self, episode=10, run_name=None):
        with self.logger as logger_run:
            self.test_eval_env(logger_run, episode)

    def test_action(self, obs):
        return self.actions(obs, np.inf, eval=True)

    def test_eval_env(self, logger_run, episode):
        # Use common test helper: (env_builder, logger_run, actions_eval_fn, episode, conv_action=None)
        record_test_fn = getattr(self, "record_test_fn", record_and_test)
        return record_test_fn(
            self.env_builder,
            logger_run,
            self.test_action,
            episode,
            conv_action=None,
        )

    # -------------------------------
    # Checkpointing scaffolding hooks
    # -------------------------------
    def _checkpoint_update_snapshot(self):
        """Default checkpoint snapshot strategy for DPG family.

        Snapshots the eval behaviour-state (mirrors eval action selection).
        Subclasses can override for custom snapshot strategies.
        """
        self.eval_snapshot = snapshot_pytree(self.get_eval_state())

        # If using SIMBA normalization, snapshot obs_rms for eval-time consistency.
        if self.simba:
            self.checkpoint_obs_rms = deepcopy(self._policy_update_obs_rms())
