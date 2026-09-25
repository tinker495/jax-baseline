import itertools
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.bulk_training import SCAN_UNROLL
from jax_baselines.core.checkpoint import make_checkpoint_scaffold, snapshot_pytree
from jax_baselines.core.checkpoint_state import QNetCheckpointState
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.env_info import get_local_env_info
from jax_baselines.core.eval import evaluate_policy, record_and_test
from jax_baselines.core.normalization import RewardNormalizer
from jax_baselines.core.replay_protocol import (
    LocalReplayNeed,
    PriorityNeed,
    ReplayBufferFactory,
    require_replay_factory,
    select_replay_device,
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
from jax_baselines.math.metrics import reduce_metrics
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


@lru_cache(maxsize=64)
def _device_update_counts(counts):
    """Pulse schedules repeat, so each distinct weight vector crosses to the device once."""
    return jax.device_put(np.asarray(counts))


class Q_Network_Family:
    _run_name = "Q_network"
    _get_actions: Callable[..., jax.Array]
    # One update: `(*state, step, key, **batch, **static) -> (*state, loss, target,
    # priorities, metrics, histograms)`.
    _train_step: Callable[..., tuple]

    supports_bulk_training = False
    # Bulk samples arrive as (chunk, batch, ...) unless the learner slices a flat chunk itself.
    flat_bulk_batches = False
    # IQN/FQF/SPR consume a PRNG key in every forward pass; the others only with param noise.
    _uses_rng = False

    def __init__(
        self,
        env_builder: Callable,
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
        reward_normalization=False,
        memory_backend: Literal["auto", "cpu", "gpu"] = "auto",
    ):
        if memory_backend not in ("auto", "cpu", "gpu"):
            raise ValueError("memory_backend must be 'auto', 'cpu', or 'gpu'")
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
        self.reward_normalization = bool(reward_normalization)

        self.get_env_setup()
        self._initial_reset = None
        self.memory_backend: Literal["cpu", "gpu"] = "cpu"
        self.memory_device = None
        if memory_backend != "cpu":
            if self.env_type == "SingleEnv":
                self._initial_reset = self.env.reset()
                initial_obs = self._initial_reset[0]
            else:
                initial_obs = self.env.current_obs()
            self.memory_device = select_replay_device(initial_obs, required=memory_backend == "gpu")
            if self.memory_device is not None:
                self.memory_backend = "gpu"
        print("memory backend : ", self.memory_backend)
        with jax.default_device(self.memory_device):
            self.reward_normalizer = (
                RewardNormalizer(self.worker_size, self.gamma)
                if self.reward_normalization
                else None
            )
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
            with jax.default_device(self.memory_device):
                self.setup_model()

    def save_params(self, path):
        if self.reward_normalizer is None:
            self.checkpoint_store.save(path, self.params)
            return
        self.checkpoint_store.save(
            path,
            QNetCheckpointState(
                params=self.params,
                reward_rms_state=self.reward_normalizer.to_state(),
            ),
        )

    def load_params(self, path):
        state = self.checkpoint_store.restore(path)
        with jax.default_device(self.memory_device):
            if self.reward_normalizer is not None:
                self.reward_normalizer.reset()
            if isinstance(state, QNetCheckpointState):
                if self.reward_normalizer is not None and state.reward_rms_state is not None:
                    self.reward_normalizer.restore(state.reward_rms_state)
                state = state.params
            self.params = self.target_params = jax.device_put(state, self.memory_device)

    def get_env_setup(self):
        (
            self.env,
            self.eval_env,
            self.observation_space,
            self.action_size,
            self.worker_size,
            self.env_type,
        ) = get_local_env_info(self.env_builder, self.num_workers, seed=self.seed)
        self.autoreset_steps = True
        if self.env_type == "VectorizedEnv":
            env_info = self.env.get_info()
            if "autoreset_steps" in env_info:
                self.autoreset_steps = env_info["autoreset_steps"]
        if not isinstance(self.autoreset_steps, bool):
            raise TypeError("Environment autoreset_steps must be a bool")
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
                memory_backend=self.memory_backend,
                device=self.memory_device,
                seed=self.seed,
            )
        )

    def setup_model(self):
        pass

    def _make_optimizer(self, learning_rate):
        # Updates pass ``diagnostics=``; plain optax transforms simply ignore it.
        return optax.with_extra_args_support(self.optimizer_factory(learning_rate))

    def train_step(self, steps, gradient_steps, logger_run=None, log_interval=None):
        return self.training_lifecycle.train(steps, gradient_steps, logger_run, log_interval)

    @property
    def _train_state(self):
        """Pytrees one update reads and returns, in `_train_step` argument order."""
        return self.params, self.target_params, self.opt_state

    @_train_state.setter
    def _train_state(self, state):
        self.params, self.target_params, self.opt_state = state

    def _train_on_batch(self, data, context):
        return QNetTrainResult.from_values(
            *self._run_update(self._compiled_update, data, diagnostics=context.diagnostics)
        )

    def _train_on_bulk(self, data, contexts):
        return QNetTrainResult.from_values(
            *self._run_update(self._compiled_bulk_scan, data, diagnostics=contexts[0].diagnostics),
            update_count=len(contexts),
        )

    def _run_update(self, update, data, **static):
        # The lifecycle already placed the batch on device; PER indexes stay where the
        # replay produced them for the priority write-back.
        batch = {key: value for key, value in data.items() if key != "indexes"}
        self._train_state, self._train_key, self._update_count, outputs = update(
            self._train_state, self._train_key, self._update_count, batch, **static
        )
        return outputs

    def _update(self, state, key, count, data, **static):
        """One `_train_step` with its PRNG key and update counter carried on device.

        `count` mirrors the host `train_steps_count`; the step sees `count + 1`,
        exactly the value the host lifecycle assigns to this update. `static` holds the
        compile-time flags (``diagnostics``, plus SPR-family ``resets``).
        `_train_step` returns ``(*state, loss, target, priorities, metrics, histograms)``.
        """
        key, subkey = jax.random.split(key)
        outputs = self._train_step(*state, count + 1, self._forward_key(subkey), **data, **static)
        return outputs[: len(state)], key, count + self._updates_in(data), outputs[len(state) :]

    def _updates_in(self, data):
        """Gradient updates one `_train_step` call performs on `data`."""
        return 1

    def _forward_key(self, key):
        return key if self.param_noise or self._uses_rng else None

    def _bulk_scan(self, state, key, count, data, diagnostics):
        def train_one(carry, batch):
            state, key, count = carry
            state, key, count, outputs = self._update(
                state, key, count, batch, diagnostics=diagnostics
            )
            return (state, key, count), outputs

        (state, key, count), (losses, targets, priorities, metrics, histograms) = jax.lax.scan(
            train_one, (state, key, count), data, unroll=SCAN_UNROLL
        )
        # Diagnostics reduce inside this compiled call; PER priorities flatten in sample order.
        loss, target, metrics, histograms = jax.tree.map(
            lambda value: jnp.mean(value, axis=0), (losses, targets, metrics, histograms)
        )
        priorities = jax.tree.map(lambda value: value.reshape(-1), priorities)
        return state, key, count, (loss, target, priorities, metrics, histograms)

    def _aggregate_train_reports(self, reports):
        if len(reports) == 1:
            return reports[-1]
        update_counts = tuple(report.update_count for report in reports)
        device_counts = _device_update_counts(update_counts)
        metric_values = {}
        metric_weights = {}
        for name in dict.fromkeys(name for report in reports for name in report.metrics):
            observations = [report for report in reports if name in report.metrics]
            metric_values[name] = tuple(report.metrics[name] for report in observations)
            metric_weights[name] = (
                device_counts
                if len(observations) == len(reports)
                else _device_update_counts(tuple(report.update_count for report in observations))
            )
        metrics, _ = reduce_metrics(metric_values, metric_weights)
        histogram_values = {
            name: tuple(report.histograms[name] for report in reports)
            for name in reports[-1].histograms
            if all(name in report.histograms for report in reports)
        }
        histograms, _ = reduce_metrics(
            histogram_values, dict.fromkeys(histogram_values, device_counts)
        )
        # Reports mirror loss/target into metrics, so the compiled reduction covers them.
        return QNetTrainReport(
            loss=metrics["loss/qloss"],
            target=(
                metrics["loss/targets"]
                if all(report.target is not None for report in reports)
                else None
            ),
            metrics=metrics,
            histograms=histograms,
            update_count=sum(update_counts),
        )

    def _compile_common_functions(self):
        """JIT compilation and device-resident carries shared by the Q-Network family."""
        self.get_q = jax.jit(self.get_q)
        self._get_actions = jax.jit(self._get_actions)
        self._compiled_behavior = jax.jit(self._behavior_step)
        self._compiled_greedy = jax.jit(self._greedy_step)
        self._compiled_update = jax.jit(self._update, static_argnames="diagnostics")
        self._compiled_bulk_scan = jax.jit(self._bulk_scan, static_argnames="diagnostics")
        # Acting, evaluation and updates each carry their own device PRNG key, so the
        # training stream does not depend on how often evaluation runs.
        self._action_key, self._eval_key, self._train_key = (next(self.key_seq) for _ in range(3))
        self._update_count = jnp.asarray(self.train_steps_count, dtype=jnp.int32)

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

    def _acting_params(self, params):
        """Arguments `_get_actions` takes as its params (FQF adds its fraction network)."""
        return params

    def _behavior_step(self, params, obses, key, behavior_steps, epsilon_table):
        """Epsilon lookup, greedy inference and exploration in one compiled call."""
        index = jnp.minimum(self._epsilon_index(behavior_steps), epsilon_table.shape[0] - 1)
        epsilon = epsilon_table[index]
        key, net_key, explore_key, random_key = jax.random.split(key, 4)
        greedy = self._get_actions(params, obses, self._forward_key(net_key))
        explore = jax.random.uniform(explore_key, greedy.shape) < epsilon
        random_actions = jax.random.randint(random_key, greedy.shape, 0, self.action_size[0])
        return jnp.where(explore, random_actions, greedy), key, behavior_steps + 1, epsilon

    def _greedy_step(self, params, obses, key):
        key, net_key = jax.random.split(key)
        return self._get_actions(params, obses, self._forward_key(net_key)), key

    def _behavior_actions(self, obs):
        # Env boundary: one explicit copy of the observation in and the actions out.
        actions, self._action_key, self._behavior_steps, self._epsilon = self._compiled_behavior(
            self._acting_params(self.get_behavior_params()),
            jax.device_put(obs),
            self._action_key,
            self._behavior_steps,
            self._epsilon_table,
        )
        return jax.device_get(actions)

    def actions(self, obs, eval_mode=False):
        """Greedy actions for evaluation and testing."""
        params = self.get_behavior_params()
        if eval_mode and self.use_checkpointing and self.ckpt.enabled:
            params = self.checkpoint_params
        actions, self._eval_key = self._compiled_greedy(
            self._acting_params(params), jax.device_put(obs), self._eval_key
        )
        return jax.device_get(actions)

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        losses, epsilon = jax.device_get((tuple(self.lossque), self._epsilon))
        description += f"loss : {np.mean(losses):.3f}"

        if not self.param_noise:
            description += f", epsilon : {epsilon:.3f}"

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
            return f"Rainbow({self.n_step} step)_" + run_name
        if self.param_noise:
            run_name = "Noisy_" + run_name
        if self.dueling_model:
            run_name = "Dueling_" + run_name
        if self.double_q:
            run_name = "Double_" + run_name
        if self.n_step_method:
            run_name = f"{self.n_step}Step_" + run_name
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
        self._exploration_steps = int(self.exploration_fraction * total_timesteps)
        # The whole exploration schedule crosses to the device once; acting indexes it with a
        # device count of behavior steps instead of uploading epsilon every step.
        table = np.asarray(self._epsilon_schedule(), dtype=np.float32)
        self._epsilon_table = jax.device_put(table)
        self._epsilon = jax.device_put(table[0])
        self._behavior_steps = jax.device_put(np.int32(0))

    def _scheduled_epsilon(self, steps):
        if self.param_noise:
            return 0.0
        epsilon = self.exploration_initial_eps
        if self._exploration_steps > 0:
            epsilon += min(max(steps / self._exploration_steps, 0.0), 1.0) * (
                self.exploration_final_eps - self.exploration_initial_eps
            )
        return epsilon

    def _epsilon_schedule(self):
        """Epsilon at each refresh point of the rollout loop, until it stops changing.

        Vectorized loops refresh before every action, at steps ``i * worker_size``. The
        single-env loops start from 1.0 and refresh after the action, only at steps past
        ``learning_starts`` that are multiples of ``train_freq``.
        """
        if self.env_type == "SingleEnv":
            first = self.learning_starts // self.train_freq + 1
            points = (self.train_freq * (first + index) for index in itertools.count())
            table = [1.0]
        else:
            points = (self.worker_size * index for index in itertools.count())
            table = []
        for steps in points:
            table.append(self._scheduled_epsilon(steps))
            if self.param_noise or steps >= self._exploration_steps:
                return table

    def _epsilon_index(self, behavior_steps):
        """Refresh points the rollout loop has passed before this behavior step."""
        if self.env_type == "SingleEnv":
            return jnp.maximum(
                (behavior_steps - 1) // self.train_freq - self.learning_starts // self.train_freq,
                0,
            )
        return behavior_steps

    def run_training_loop(self, ctx):
        off_policy_loop(self, ctx)

    # -------------------------------
    # Rollout seam (RolloutSpec wiring)
    # -------------------------------
    def _bind_loss_window(self, window):
        self.lossque = window

    def _single_action_selection(self, obs, steps):
        actions = self._behavior_actions(obs)
        return ActionSelection(env_action=actions[0][0].item(), store_action=actions[0])

    def _vector_action_selection(self, obs, steps):
        actions = self._behavior_actions(obs)
        return ActionSelection(env_action=actions, store_action=actions)

    def _write_ckpt_residual(self, value):
        self._ckpt_update_residual = value

    def make_rollout_spec(self, ctx):
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)

        def train(steps, gradient_steps):
            loss = self.train_step(steps, gradient_steps, ctx.logger_run, ctx.log_interval)
            ctx.progress.update_steps += gradient_steps
            return loss

        pulse = CheckpointTrainPulse(
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            worker_size=self.worker_size,
            train=train,
            record_loss=lambda loss: self.lossque.append(loss),
            read_residual=lambda: self._ckpt_update_residual,
            write_residual=self._write_ckpt_residual,
        )
        spec = RolloutSpec(
            env=self.env,
            progress=ctx.progress,
            logger_run=ctx.logger_run,
            replay_buffer=self.replay_buffer,
            learning_starts=self.learning_starts,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            eval_freq=ctx.eval_freq,
            worker_size=self.worker_size,
            single_action=self._single_action_selection,
            vector_action=self._vector_action_selection,
            # Epsilon follows the device behavior-step count (see _epsilon_schedule).
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
            memory_device=self.memory_device,
            autoreset_steps=self.autoreset_steps,
            initial_reset=self._initial_reset,
        )
        self._initial_reset = None
        return spec

    def eval(self, ctx, steps):
        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            lambda obs: self.actions(obs, eval_mode=True),
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
            self.actions,
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
