from collections.abc import Callable
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.bulk_training import SCAN_UNROLL
from jax_baselines.core.checkpoint import make_checkpoint_scaffold, snapshot_pytree
from jax_baselines.core.checkpoint_state import CheckpointState
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.env_info import get_local_env_info
from jax_baselines.core.eval import evaluate_policy, record_and_test
from jax_baselines.core.normalization import RewardNormalizer, RunningMeanStd
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
from jax_baselines.DDPG.metrics import stochastic_actor_metrics
from jax_baselines.DDPG.training import DPGTrainingLifecycle, DPGTrainReport
from jax_baselines.math.metrics import reduce_metrics
from jax_baselines.optim import (
    OptimizerFactory,
    optimizer_metrics,
    require_optimizer_factory,
    track_optimizer,
)


class UpdateFlags(NamedTuple):
    """Static switches of one compiled update, decided on the host from the update schedule.

    A device predicate (`lax.cond`) would make the GPU copy it to the host every update.
    """

    actor: bool
    reset: bool
    diagnostics: bool


def merge_actor_metrics(metrics, actor_metrics, actor_updated):
    """Per-update metrics and counts; actor metrics count only on actor-update steps."""
    counts = {name: jnp.asarray(1) for name in metrics}
    counts.update(dict.fromkeys(actor_metrics, jnp.asarray(actor_updated, dtype=jnp.int32)))
    return {**metrics, **actor_metrics}, counts


@jax.jit(static_argnums=1)
def _uniform_actions(key, shape):
    key, sample_key = jax.random.split(key)
    return jax.random.uniform(sample_key, shape, minval=-1.0, maxval=1.0), key


class Deteministic_Policy_Gradient_Family:
    _ent_coef: str | float
    ent_coef_learning_rate: float | optax.Schedule
    target_entropy: float
    _run_name = "DPG_network"
    # One pure update: `(state, key, step, flags, **batch) -> (state, (priorities, metrics,
    # counts))`. `step` is the device 1-based update count and `flags` the static
    # `UpdateFlags`; `metrics` always holds `loss/qloss`, diagnostics only when
    # `flags.diagnostics`. Every update of a chunk must return the same output structure.
    _train_step: Callable[..., tuple]
    # Behavior actions `(state, obses, key, ...) -> (actions, key, ...)` and eval actions
    # `(state, obses) -> actions`.
    _get_actions: Callable[..., tuple]
    _get_eval_actions: Callable[..., jax.Array]

    supports_bulk_training = True
    # Actor updates on 1-based update steps where (step - offset) % period == 0.
    _actor_schedule = (1, 0)

    def __init__(
        self,
        env_builder: Callable,
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
        obs_rms_norm: bool = False,
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
        self.obs_rms_norm = obs_rms_norm
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.optimizer = optax.with_extra_args_support(self._make_optimizer(self.learning_rate))
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

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            with jax.default_device(self.memory_device):
                self.setup_model()
        # Drawn after setup_model so model init keeps its key. Action/update PRNG keys and the
        # update counter stay on device and are carried through the compiled calls.
        with jax.default_device(self.memory_device):
            self._action_key = next(self.key_seq)
            self._train_key = next(self.key_seq)
            self._update_count = jnp.zeros((), dtype=jnp.int32)
        self._compiled_actions = jax.jit(self._get_actions)
        self._compiled_eval_actions = jax.jit(self._get_eval_actions)
        self._compiled_update = jax.jit(self._counted_update, static_argnums=2)
        self._compiled_updates = jax.jit(self._planned_updates, static_argnums=2)

        self.eval_snapshot = None
        if self.obs_rms_norm:
            with jax.default_device(self.memory_device):
                self.obs_rms = RunningMeanStd(shapes=self.observation_space)
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
            obs_rms_state=self.obs_rms.to_state() if self.obs_rms_norm else None,
            action_obs_rms_state=(
                self.action_obs_rms.to_state()
                if (self.obs_rms_norm and self.action_obs_rms is not None)
                else None
            ),
            checkpoint_obs_rms_state=(
                self.checkpoint_obs_rms.to_state()
                if (self.obs_rms_norm and self.checkpoint_obs_rms is not None)
                else None
            ),
            reward_rms_state=(
                self.reward_normalizer.to_state() if self.reward_normalizer is not None else None
            ),
        )

    def _restore_checkpoint_state(self, state: CheckpointState):
        self.load_checkpoint_params(jax.device_put(state.params, self.memory_device))
        self.train_steps_count = int(np.asarray(state.train_steps_count).item())
        self._update_count = jax.device_put(np.int32(self.train_steps_count), self.memory_device)
        self._ckpt_update_residual = float(np.asarray(state.ckpt_residual).item())
        self.ckpt.from_state(state.controller_state)
        self.eval_snapshot = jax.device_put(state.eval_snapshot, self.memory_device)

        with jax.default_device(self.memory_device):
            if self.obs_rms_norm:
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

            if self.reward_normalizer is not None:
                self.reward_normalizer.reset()
                if state.reward_rms_state is not None:
                    self.reward_normalizer.restore(state.reward_rms_state)

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
                action_shape_or_n=self.action_size,
                worker_size=self.worker_size,
                n_step=self.n_step if self.n_step_method else 1,
                gamma=self.gamma,
                priority=priority,
                memory_backend=self.memory_backend,
                device=self.memory_device,
                seed=self.seed,
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

        self.ent_coef_optimizer = track_optimizer(
            optax.adam(self.ent_coef_learning_rate), self.ent_coef_learning_rate
        )
        self.opt_ent_coef_state = self.ent_coef_optimizer.init(self.log_ent_coef)

    def _train_ent_coef(self, log_coef, opt_state, log_prob, diagnostics):
        def loss(log_ent_coef):
            entropy = -jax.lax.stop_gradient(log_prob)
            return jnp.mean(jnp.exp(log_ent_coef) * (entropy - self.target_entropy))

        ent_coef_loss, grad = jax.value_and_grad(loss)(log_coef)
        updates, opt_state = self.ent_coef_optimizer.update(
            grad, opt_state, log_coef, diagnostics=diagnostics
        )
        metrics = (
            {"loss/ent_coef_loss": ent_coef_loss, **optimizer_metrics(opt_state, "ent_coef")}
            if diagnostics
            else {}
        )
        return optax.apply_updates(log_coef, updates), opt_state, metrics

    def _skipped_actor_metrics(self, ent_coef, opt_policy_state, opt_ent_coef_state):
        """Zero placeholders (count 0) so skipped actor steps keep the metric structure."""
        names = (
            "loss/actor_loss",
            *stochastic_actor_metrics(
                jnp.zeros(1), jnp.zeros(1), 0.0, ent_coef, self.target_entropy
            ),
            *optimizer_metrics(opt_policy_state, "actor"),
        )
        if self.auto_entropy:
            names += ("loss/ent_coef_loss", *optimizer_metrics(opt_ent_coef_state, "ent_coef"))
        return dict.fromkeys(names, jnp.asarray(0.0))

    def train_step(self, steps, gradient_steps, logger_run=None, log_interval=None):
        return self.training_lifecycle.train(steps, gradient_steps, logger_run, log_interval)

    def prepare_run(self, total_timesteps):
        pass

    def run_training_loop(self, ctx):
        off_policy_loop(self, ctx)

    @property
    def _train_state(self):
        """Tuple of learner state threaded through `_train_step`; subclasses add a setter."""
        raise NotImplementedError

    def _update_flags(self, step, diagnostics):
        period, offset = self._actor_schedule
        return UpdateFlags(
            actor=(step - offset) % period == 0,
            reset=self.scaled_by_reset and step % self.reset_freq == 0,
            diagnostics=diagnostics,
        )

    def _update_plan(self, first_step, count, diagnostics):
        """Run-length encode a chunk's per-update flags into `(pattern, repeats)` segments.

        Patterns are one actor period long, so a chunk becomes a scan over the phase-rotated
        pattern plus a tail of `count % period` updates; a reset is its own segment.
        """
        period, _ = self._actor_schedule
        flags = [self._update_flags(first_step + index, diagnostics) for index in range(count)]
        plan = []
        for start in range(0, count, period):
            pattern = tuple(flags[start : start + period])
            if plan and plan[-1][0] == pattern:
                plan[-1] = (pattern, plan[-1][1] + 1)
            else:
                plan.append((pattern, 1))
        return tuple(plan)

    def _counted_update(self, carry, batch, flags):
        state, key, count = carry
        key, update_key = jax.random.split(key)
        count = count + 1
        state, outputs = self._train_step(state, update_key, count, flags, **batch)
        return (state, key, count), outputs

    def _update_group(self, carry, batches, pattern):
        outputs = []
        for index, flags in enumerate(pattern):
            carry, output = self._counted_update(
                carry, jax.tree.map(lambda value: value[index], batches), flags
            )
            outputs.append(output)
        return carry, jax.tree.map(lambda *values: jnp.stack(values), *outputs)

    def _planned_updates(self, carry, batches, plan):
        """Run a chunk in update order: each segment scans `repeats` groups of its pattern."""
        outputs = []
        start = 0
        for pattern, repeats in plan:
            size = len(pattern) * repeats
            segment = jax.tree.map(
                lambda value: value[start : start + size].reshape(
                    repeats, len(pattern), *value.shape[1:]
                ),
                batches,
            )
            carry, output = jax.lax.scan(
                partial(self._update_group, pattern=pattern),
                carry,
                segment,
                unroll=min(repeats, max(1, SCAN_UNROLL // len(pattern))),
            )
            outputs.append(
                jax.tree.map(lambda value: value.reshape(size, *value.shape[2:]), output)
            )
            start += size
        priorities, metrics, metric_counts = jax.tree.map(
            lambda *values: jnp.concatenate(values), *outputs
        )
        if priorities is not None:
            priorities = priorities.reshape(-1)
        return carry, (priorities, *reduce_metrics(metrics, metric_counts))

    def _run_update(self, compiled, data, schedule):
        # The lifecycle already placed the batch on device; `indexes` only feeds the
        # host-side priority write-back.
        batch = {name: value for name, value in data.items() if name != "indexes"}
        (self._train_state, self._train_key, self._update_count), (
            priorities,
            metrics,
            metric_counts,
        ) = compiled((self._train_state, self._train_key, self._update_count), batch, schedule)
        return DPGTrainReport(metrics, metric_counts, priorities)

    def _train_on_batch(self, data, flags):
        return self._run_update(self._compiled_update, data, flags)

    def _train_on_bulk(self, data, plan):
        return self._run_update(self._compiled_updates, data, plan)

    def _aggregate_train_reports(self, reports):
        if len(reports) == 1:
            return reports[-1]
        names = reports[0].metrics
        # Every count is a device array, so the reduction is one compiled call with no transfer.
        return DPGTrainReport(
            *reduce_metrics(
                {name: tuple(report.metrics[name] for report in reports) for name in names},
                {name: tuple(report.metric_counts[name] for report in reports) for name in names},
            )
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
        # Actions stay on device only for envs that return device observations; a host env
        # gets them through one explicit download.
        device_env = isinstance(next(iter(obs.values())), jax.Array)
        obs = self._normalize_action_observation(obs, eval, steps)
        if not eval and steps <= self.learning_starts:
            actions = self._random_warmup_actions()
        else:
            state = self._select_action_state(eval, steps)
            actions = self._policy_action_from_state(state, jax.device_put(obs), eval, steps)
        return actions if device_env else jax.device_get(actions)

    def _random_warmup_actions(self):
        shape = (self.worker_size, self.action_size[0])
        if self.memory_backend == "cpu":
            return np.random.uniform(-1.0, 1.0, size=shape)
        actions, self._action_key = _uniform_actions(self._action_key, shape)
        return actions

    def _select_action_state(self, eval, steps):
        if eval and self.use_checkpointing and self.ckpt.enabled and self.eval_snapshot is not None:
            return self.eval_snapshot
        return self.get_behavior_state()

    def _policy_action_from_state(self, state, obs, eval, steps):
        """Final env action: inference, exploration noise and clipping in one compiled call."""
        if eval:
            return self._compiled_eval_actions(state, obs)
        actions, self._action_key = self._compiled_actions(state, obs, self._action_key)
        return actions

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(jax.device_get(tuple(self.lossque))):.3f}"
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
        if self.obs_rms_norm:
            run_name = "ObsRMS_" + run_name
        if self.n_step_method:
            run_name = f"{self.n_step}Step_" + run_name
        if self.prioritized_replay:
            run_name = run_name + "+PER"
        return run_name

    def _normalize_action_observation(self, obs, eval, steps):
        if not self.obs_rms_norm:
            return obs
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
        if eval or steps == np.inf:
            return rms.normalize(obs)
        # Update and normalize in one compiled call; a snapshot keeps its frozen statistics.
        return self.obs_rms.observe(obs, None if rms is self.obs_rms else rms)

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
        if self.obs_rms_norm:
            self.action_obs_rms = self.obs_rms.snapshot()

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
            post_pulse=self._snapshot_action_normalizer,
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
            lambda obs: self.actions(obs, steps, eval=True),
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

        # If using observation RMS normalization, snapshot obs_rms for eval-time consistency.
        if self.obs_rms_norm:
            self.checkpoint_obs_rms = self._policy_update_obs_rms().snapshot()
