from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from itertools import chain, pairwise
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.checkpoint_state import ACCheckpointState
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.env_info import get_local_env_info, infer_action_meta
from jax_baselines.core.env_protocols import (
    batch_observation,
    log_environment_metrics,
    vector_autoreset_mask,
)
from jax_baselines.core.epoch_buffer import EpochBuffer
from jax_baselines.core.eval import (
    _normalize_action_for_step,
    evaluate_policy,
    record_and_test,
)
from jax_baselines.core.normalization import (
    RunningMeanStd,
    normalize_empirical_observation,
    observe_empirical_observations,
)
from jax_baselines.core.replay_protocol import select_replay_device
from jax_baselines.core.rollout_stats import EpisodeTracker, device_episode_step
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.core.training_session import TrainingSession
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


@jax.jit
def _sample_continuous(mu, std, key):
    return mu + std * jax.random.normal(key, mu.shape, dtype=mu.dtype)


@jax.jit
def _sample_discrete(prob, key):
    return jax.random.categorical(key, jnp.log(prob), axis=-1)[:, None]


def _categorical_policy(logits):
    prob = jnp.clip(jax.nn.softmax(logits, axis=-1), 1e-8, 1.0)
    return prob / jnp.sum(prob, axis=-1, keepdims=True)


@jax.jit
def _categorical_log_prob(prob, action):
    return jnp.log(jnp.take_along_axis(prob, action.astype(jnp.int32), axis=1))


def _continuous_log_prob(prob, action):
    mu, log_std = prob
    std = jnp.exp(log_std)
    return -(
        0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
        + jnp.sum(log_std, axis=-1, keepdims=True)
        + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
    )


@jax.jit
def _continuous_old_policy(policy, actions):
    mu, _, log_std = policy
    old_policy = (mu, jnp.broadcast_to(log_std, mu.shape))
    return _continuous_log_prob(old_policy, actions), old_policy


@jax.jit
def _device_rollout_step(state, transitions, rewards, terminateds, truncateds, autoreset):
    """Episode stats and live-transition count for one device vector step, compiled once."""
    transitions = transitions + jnp.count_nonzero(~state[2])
    state, rewards, terminateds, completed = device_episode_step(
        state, rewards, terminateds, truncateds, autoreset
    )
    return state, transitions, rewards, terminateds, completed


class Actor_Critic_Policy_Gradient_Family:
    _run_name = "A2C"
    _store_old_policy = False
    actor: Callable
    _get_actions: Callable
    _train_step: Callable[..., tuple]
    logger: AbstractContextManager

    def __init__(
        self,
        env_builder,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory: OptimizerFactory | None = None,
        lr_annealing=False,
        checkpoint_store: CheckpointStore | None = None,
        obs_rms_norm: bool = False,
        memory_backend: Literal["auto", "cpu", "gpu"] = "auto",
    ):
        if memory_backend not in ("auto", "cpu", "gpu"):
            raise ValueError("memory_backend must be 'auto', 'cpu', or 'gpu'")
        if use_entropy_adv_shaping and (not np.isfinite(ent_coef) or ent_coef < 0):
            raise ValueError("entropy shaping requires finite ent_coef >= 0")
        if use_entropy_adv_shaping and (
            not np.isfinite(entropy_adv_shaping_kappa) or entropy_adv_shaping_kappa <= 1
        ):
            raise ValueError("entropy shaping requires finite entropy_adv_shaping_kappa > 1")
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self._last_log_step = 0
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        set_global_seeds(self.seed)
        self.key_seq = key_gen(self.seed)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.log_dir = log_dir
        self.use_entropy_adv_shaping = use_entropy_adv_shaping
        self.entropy_adv_shaping_kappa = entropy_adv_shaping_kappa
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.lr_annealing = lr_annealing
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)

        self.actor_params = None
        self.critic_params = None
        self.rollout_tracker = None
        self.optimizer = self._make_optimizer(self.learning_rate)

        self.get_env_setup()
        self._initial_reset: tuple[dict, dict] | None = None
        self.memory_device = None
        if memory_backend != "cpu":
            if self.env_type == "SingleEnv":
                self._initial_reset = self.env.reset()
                initial_obs = self._initial_reset[0]
            else:
                initial_obs = self.env.current_obs()
            self.memory_device = select_replay_device(initial_obs, required=memory_backend == "gpu")
        self.memory_backend: Literal["cpu", "gpu"] = (
            "gpu" if self.memory_device is not None else "cpu"
        )
        print("memory backend : ", self.memory_backend)
        self.obs_rms_norm = obs_rms_norm
        with jax.default_device(self.memory_device):
            self.obs_rms = (
                RunningMeanStd(epsilon=0.0, shapes=self.observation_space) if obs_rms_norm else None
            )
        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        with jax.default_device(self.memory_device):
            if self._init_setup_model:
                self.setup_model()
            # Behavior-policy PRNG carry; drawn after model init so initial params are unchanged.
            self._action_key = next(self.key_seq)

    def save_params(self, path):
        self.checkpoint_store.save(
            path,
            ACCheckpointState(
                actor_params=self.actor_params,
                critic_params=self.critic_params,
                obs_rms_state=self.obs_rms.to_state() if self.obs_rms is not None else None,
            ),
        )

    def load_params(self, path):
        state = self.checkpoint_store.restore(path)
        if not isinstance(state, ACCheckpointState):
            raise TypeError("Expected ACCheckpointState with observation-normalization state")
        with jax.default_device(self.memory_device):
            obs_rms = (
                RunningMeanStd.from_state(state.obs_rms_state)
                if state.obs_rms_state is not None
                else None
            )
        if obs_rms is not None and (
            obs_rms.means.keys() != self.observation_space.keys()
            or any(
                obs_rms.means[key].shape != tuple(shape)
                for key, shape in self.observation_space.items()
            )
        ):
            raise ValueError("Checkpoint observation statistics do not match the environment")
        self.actor_params = jax.device_put(state.actor_params, self.memory_device)
        self.critic_params = jax.device_put(state.critic_params, self.memory_device)
        self.obs_rms = obs_rms
        self.obs_rms_norm = obs_rms is not None

    def get_memory_setup(self):
        self.buffer = EpochBuffer(
            self.batch_size,
            self.observation_space,
            self.worker_size,
            [1] if self.action_type == "discrete" else self.action_size,
            memory_backend=self.memory_backend,
            memory_device=self.memory_device,
        )

    def get_env_setup(self):
        # Use helper to standardize environment info
        (
            self.env,
            self.eval_env,
            self.observation_space,
            self.action_size,
            self.worker_size,
            self.env_type,
            action_type,
        ) = get_local_env_info(
            self.env_builder,
            self.num_workers,
            seed=self.seed,
            include_action_type=True,
        )

        self.action_type, self.conv_action = infer_action_meta(action_type)

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        if self.action_type == "discrete":
            self._get_actions = self._get_actions_discrete
            self.get_logprob = self.get_logprob_discrete
            self._actor_loss = self._actor_loss_discrete
        elif self.action_type == "continuous":
            self._get_actions = self._get_actions_continuous
            self.get_logprob = self.get_logprob_continuous
            self._actor_loss = self._actor_loss_continuous

    def setup_model(self):
        pass

    def train_step(self, steps, logger_run=None, log_interval=None):
        raise NotImplementedError

    def _metrics_due(self, steps, logger_run, log_interval):
        """Host-side schedule for the static ``diagnostics`` flag of the next update.

        Only updates whose metrics are logged compute diagnostics; the others return
        just the critic loss the progress bar reads.
        """
        interval = self.log_interval if log_interval is None else log_interval
        return bool(logger_run) and steps - self._last_log_step >= interval

    def _log_train_metrics(self, metrics, steps, logger_run):
        self._last_log_step = steps
        for name, value in jax.device_get(metrics).items():
            logger_run.log_metric(name, value, steps)

    def _get_actions_discrete(self, actor_params, obses, key=None) -> jnp.ndarray:
        return _categorical_policy(self.actor(actor_params, key, convert_normalized_obs(obses)))

    def _get_actions_continuous(
        self, actor_params, obses, key=None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mu, log_std = self.actor(actor_params, key, convert_normalized_obs(obses))
        return mu, jnp.exp(log_std), log_std

    def sample_actions(self, obs, device_env):
        """Run the compiled behavior-action stage once for one env step.

        Returns ``(actions, step_actions, old_policy)``: unclipped actions and the
        old-policy snapshot for the rollout buffer, and the env-ready actions. The
        observation enters with one ``device_put``; results leave with one
        ``device_get`` unless both buffer and env keep device arrays.
        """
        actions, step_actions, old_policy, self._action_key = self._sample_actions(
            self.actor_params, jax.device_put(obs, self.memory_device), self._action_key
        )
        if self.memory_backend == "cpu":
            return jax.device_get((actions, step_actions, old_policy))
        return actions, step_actions if device_env else jax.device_get(step_actions), old_policy

    @jax.jit(static_argnums=0)
    def _sample_actions(self, actor_params, obs, key):
        key, sample_key = jax.random.split(key)
        policy = self._get_actions(actor_params, obs)
        if self.action_type == "discrete":
            actions = _sample_discrete(policy, sample_key)
            old_policy = (_categorical_log_prob(policy, actions), policy)
            step_actions = actions
        else:
            mu, std, _ = policy
            actions = _sample_continuous(mu, std, sample_key)
            old_policy = _continuous_old_policy(policy, actions)
            step_actions = self.conv_action(actions)
        return actions, step_actions, old_policy if self._store_old_policy else None, key

    def _policy_actions(self, obs):
        """Deterministic env-ready actions for evaluation and testing.

        Actions stay on device only for envs that return device observations.
        """
        model_obs = normalize_empirical_observation(obs, self.obs_rms)
        actions = self._eval_actions(
            self.actor_params, jax.device_put(model_obs, self.memory_device)
        )
        return (
            actions if isinstance(next(iter(obs.values())), jax.Array) else jax.device_get(actions)
        )

    @jax.jit(static_argnums=0)
    def _eval_actions(self, actor_params, obs):
        policy = self._get_actions(actor_params, obs)
        if self.action_type == "discrete":
            return jnp.argmax(policy, axis=1, keepdims=True)
        return self.conv_action(policy[0])

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = _categorical_policy(prob)
        log_prob = _categorical_log_prob(prob, action)
        return (prob, log_prob) if out_prob else log_prob

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        log_prob = _continuous_log_prob(prob, action)
        return (prob, log_prob) if out_prob else log_prob

    def _actor_loss_continuous(self):
        raise NotImplementedError

    def _actor_loss_discrete(self):
        raise NotImplementedError

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(jax.device_get(tuple(self.lossque))):.3f}"

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
        return run_name

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

    def _optimizer_updates_per_train_step(self):
        rollout_size = max(1, int(self.batch_size) * int(self.worker_size))
        minibatch_size = getattr(self, "minibatch_size", None)
        minibatches = (
            max(1, int(np.ceil(rollout_size / int(minibatch_size))))
            if minibatch_size is not None
            else 1
        )
        return max(1, int(getattr(self, "epoch_num", 1)) * minibatches)

    def _lr_annealing_transition_steps(self, total_timesteps):
        # Optax schedules tick per optimizer update, so translate env timesteps
        # through the current on-policy rollout/epoch/minibatch update geometry.
        rollout_size = max(1, int(self.batch_size) * int(self.worker_size))
        train_steps = max(1, int(total_timesteps) // rollout_size)
        return train_steps * self._optimizer_updates_per_train_step()

    def prepare_run(self, total_timesteps):
        if not self.lr_annealing or self.actor_params is None:
            return

        schedule = optax.linear_schedule(
            init_value=self.learning_rate,
            end_value=0.0,
            transition_steps=self._lr_annealing_transition_steps(total_timesteps),
        )
        self.optimizer = self._make_optimizer(schedule)
        with jax.default_device(self.memory_device):
            self.actor_opt_state = self.optimizer.init(self.actor_params)
            self.critic_opt_state = self.optimizer.init(self.critic_params)

    def run_training_loop(self, ctx):
        with jax.default_device(self.memory_device):
            if self.env_type == "SingleEnv":
                self.learn_SingleEnv(ctx)
            if self.env_type == "VectorizedEnv":
                self.learn_VectorizedEnv(ctx)

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

    def learn_SingleEnv(self, ctx):
        obs, _ = self.env.reset() if self._initial_reset is None else self._initial_reset
        self._initial_reset = None
        device_env = isinstance(next(iter(obs.values())), jax.Array)
        obs = normalize_empirical_observation(batch_observation(obs), self.obs_rms)
        self.lossque = deque(maxlen=10)
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        eval_result = None
        score = 0.0
        eplen = 0
        steps = 0
        last_log_step = 0
        for steps in ctx.pbar:
            actions, step_actions, old_policy = self.sample_actions(obs, device_env)
            next_obs, reward, terminated, truncated, _ = self.env.step(
                _normalize_action_for_step(step_actions)
            )
            ctx.progress.env_steps += 1
            log_due = steps - last_log_step >= ctx.log_interval
            log_environment_metrics(self.env, ctx.logger_run, steps, flush=log_due)
            if log_due:
                last_log_step = steps
            next_obs = batch_observation(next_obs)
            if terminated or truncated:
                next_obs = {key: value.copy() for key, value in next_obs.items()}
                action_observation, _ = self.env.reset()
                action_observation = batch_observation(action_observation)
            else:
                action_observation = next_obs
            # Timeout bootstraps must not incorporate the next episode's reset sample.
            next_obs, action_observation = observe_empirical_observations(
                self.obs_rms, next_obs, action_observation
            )
            self.buffer.add(
                obs,
                actions,
                [reward],
                next_obs,
                [terminated],
                [truncated],
                old_policy=old_policy,
            )
            score += float(reward)
            eplen += 1
            obs = action_observation

            if terminated or truncated:
                self.rollout_tracker.record(
                    steps,
                    episode_reward=score,
                    episode_length=eplen,
                    timeout=float(truncated),
                )
                score = 0.0
                eplen = 0

            if (steps + 1) % self.batch_size == 0:
                loss = self.train_step(
                    steps, logger_run=ctx.logger_run, log_interval=ctx.log_interval
                )
                ctx.progress.update_steps += self._optimizer_updates_per_train_step()
                self.lossque.append(loss)

            if steps % ctx.eval_freq == 0:
                eval_result = self.eval(ctx, steps)

            if log_due and eval_result is not None and len(self.lossque) > 0:
                ctx.pbar.set_description(self.description(eval_result))
            if log_due:
                ctx.progress.log(ctx.logger_run, steps)
        log_environment_metrics(self.env, ctx.logger_run, steps)
        ctx.progress.log(ctx.logger_run, steps)

    def learn_VectorizedEnv(self, ctx):
        raw_obs = self.env.current_obs()
        device_rollout = self.memory_backend == "gpu" and isinstance(
            next(iter(raw_obs.values())), jax.Array
        )
        if device_rollout:
            # Episode state and the live-transition count stay on device between log downloads.
            device_state, no_transitions = jax.device_put(
                (
                    (
                        np.zeros(self.worker_size, dtype=np.float32),
                        np.zeros(self.worker_size, dtype=np.int32),
                        np.zeros(self.worker_size, dtype=bool),
                    ),
                    np.zeros((), dtype=np.int32),
                ),
                self.memory_device,
            )
            transitions = no_transitions
        completed_steps = []
        completed_rows = []
        self.lossque = deque(maxlen=10)
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        eval_result = None
        scores = np.zeros([self.worker_size], dtype=np.float64)
        eplens = np.zeros([self.worker_size], dtype=np.int32)
        # Workers that ended an episode last step emit an autoreset dummy step
        # (action ignored, reward 0, fresh obs). On-policy rollouts have a fixed
        # per-worker length, so the dummy can't be dropped; instead flag it
        # terminal so it contributes a zero-value target and never bridges the
        # two episodes in the return. prev_done chains off the *real* env dones,
        # and the same mask keeps the dummy out of the rollout episode stats.
        prev_done = np.zeros(self.worker_size, dtype=bool)

        # Pipeline the async env between updates. At an update boundary, delay
        # the next send until train_step finishes so the new rollout cannot start
        # with an action sampled from the previous policy. Pair each step with its
        # successor so the final iteration does not leave an env result pending.
        obs = normalize_empirical_observation(raw_obs, self.obs_rms)
        actions, step_actions, old_policy = self.sample_actions(obs, device_rollout)
        end = object()
        first = True
        steps = 0
        last_log_step = 0
        for steps, next_step in pairwise(chain(ctx.pbar, (end,))):
            if first:
                self.env.step(step_actions)
                first = False
            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()
            log_due = steps - last_log_step >= ctx.log_interval
            log_environment_metrics(
                self.env,
                ctx.logger_run,
                steps,
                flush=log_due,
            )
            if log_due:
                last_log_step = steps
            action_observation = self.env.current_obs()
            # Autoreset observations belong to the next action, not this successor.
            next_obses, action_observation = observe_empirical_observations(
                self.obs_rms, next_obses, action_observation
            )

            train_due = (steps + self.worker_size) % (self.batch_size * self.worker_size) == 0
            if not train_due and next_step is not end:
                # Keep the async overlap except when train_step changes the policy.
                next_actions, next_step_actions, next_old_policy = self.sample_actions(
                    action_observation, device_rollout
                )
                self.env.step(next_step_actions)

            if device_rollout:
                device_state, transitions, rewards, terminateds, completed = _device_rollout_step(
                    device_state,
                    transitions,
                    rewards,
                    terminateds,
                    truncateds,
                    vector_autoreset_mask(self.env, terminateds, truncateds, infos),
                )
                self.buffer.add(
                    obs,
                    actions,
                    rewards,
                    next_obses,
                    terminateds,
                    truncateds,
                    old_policy=old_policy,
                )
                completed_steps.append(steps)
                completed_rows.append(completed)
                if log_due or next_step is end:
                    # Episode stats are only read when logging: one stack, one download.
                    with jax.profiler.TraceAnnotation("rollout.metrics"):
                        episode_rows, transition_count = jax.device_get(
                            (jnp.stack(completed_rows), transitions)
                        )
                    transitions = no_transitions
                    ctx.progress.env_steps += int(transition_count)
                    for time_idx, worker_idx in np.argwhere(episode_rows[..., 0]):
                        row = episode_rows[time_idx, worker_idx]
                        self.rollout_tracker.record(
                            completed_steps[time_idx],
                            episode_reward=float(row[1]),
                            episode_length=int(row[2]),
                            timeout=float(row[3]),
                        )
                    completed_steps.clear()
                    completed_rows.clear()
            else:
                done = np.logical_or(terminateds, truncateds)
                autoreset = vector_autoreset_mask(self.env, terminateds, truncateds, infos)
                active = ~prev_done
                ctx.progress.env_steps += int(active.sum())
                scores[active] += rewards[active]
                eplens[active] += 1

                if prev_done.any():
                    # Flag the dummy step terminal AND zero its reward so it is fully
                    # inert (zero-value target, no episode bridge), independent of
                    # whatever the env reports on the discarded autoreset step.
                    terminateds = np.where(prev_done, True, terminateds)
                    rewards = np.where(prev_done, np.float32(0.0), rewards)
                self.buffer.add(
                    obs,
                    actions,
                    rewards,
                    next_obses,
                    terminateds,
                    truncateds,
                    old_policy=old_policy,
                )

                for idx in np.where(done & active)[0]:
                    self.rollout_tracker.record(
                        steps,
                        episode_reward=float(scores[idx]),
                        episode_length=int(eplens[idx]),
                        timeout=float(truncateds[idx]),
                    )
                    scores[idx] = 0.0
                    eplens[idx] = 0

                prev_done = done & autoreset & active

            if train_due:
                loss = self.train_step(
                    steps, logger_run=ctx.logger_run, log_interval=ctx.log_interval
                )
                ctx.progress.update_steps += self._optimizer_updates_per_train_step()
                self.lossque.append(loss)
                if next_step is not end:
                    next_actions, next_step_actions, next_old_policy = self.sample_actions(
                        action_observation, device_rollout
                    )
                    self.env.step(next_step_actions)

            if next_step is not end:
                # The successor stored in replay may be a terminal observation;
                # the action just sent belongs to the env's current observation.
                obs = action_observation
                actions = next_actions
                old_policy = next_old_policy

            if steps % ctx.eval_freq == 0:
                eval_result = self.eval(ctx, steps)

            if log_due and eval_result is not None and len(self.lossque) > 0:
                ctx.pbar.set_description(self.description(eval_result))
            if log_due:
                ctx.progress.log(ctx.logger_run, steps)
        log_environment_metrics(self.env, ctx.logger_run, steps)
        ctx.progress.log(ctx.logger_run, steps)

    def eval(self, ctx, steps):
        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            self._policy_actions,
            logger_run=ctx.logger_run,
            steps=steps,
        )

    def test(self, episode=10):
        with self.logger as logger_run:
            self.test_eval_env(logger_run, episode)

    def test_eval_env(self, logger_run, episode):
        record_test_fn = getattr(self, "record_test_fn", record_and_test)
        return record_test_fn(
            self.env_builder,
            logger_run,
            self._policy_actions,
            episode,
        )
