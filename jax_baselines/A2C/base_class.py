from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.env_info import get_local_env_info, infer_action_meta
from jax_baselines.core.env_protocols import (
    vector_autoreset_mask,
    vector_real_reset_mask,
)
from jax_baselines.core.epoch_buffer import EpochBuffer
from jax_baselines.core.eval import (
    _normalize_action_for_step,
    evaluate_policy,
    extract_lives,
    extract_original_reward,
    extract_vector_original_rewards,
    record_and_test,
)
from jax_baselines.core.rollout_stats import EpisodeTracker
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.core.training_session import TrainingSession
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class Actor_Critic_Policy_Gradient_Family(object):
    _run_name = "A2C"

    def __init__(
        self,
        env_builder,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        val_coef=0.2,
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

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.log_dir = log_dir
        self.use_entropy_adv_shaping = use_entropy_adv_shaping
        self.entropy_adv_shaping_kappa = entropy_adv_shaping_kappa
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.lr_annealing = lr_annealing
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)

        self.params = None
        self.rollout_tracker = None
        self.optimizer = self._make_optimizer(self.learning_rate)

        self.get_env_setup()
        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()

    def save_params(self, path):
        self.checkpoint_store.save(path, self.params)

    def load_params(self, path):
        self.params = self.checkpoint_store.restore(path)

    def get_memory_setup(self):
        self.buffer = EpochBuffer(
            self.batch_size,
            self.observation_space,
            self.worker_size,
            [1] if self.action_type == "discrete" else self.action_size,
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
            self._loss = self._loss_discrete
            self.actions = self.action_discrete
        elif self.action_type == "continuous":
            self._get_actions = self._get_actions_continuous
            self.get_logprob = self.get_logprob_continuous
            self._loss = self._loss_continuous
            self.actions = self.action_continuous

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions_discrete(self, params, obses, key=None) -> jnp.ndarray:
        prob = jax.nn.softmax(
            self.actor(params, key, self.preproc(params, key, convert_jax(obses))),
            axis=1,
        )
        return prob

    def _get_actions_continuous(self, params, obses, key=None) -> jnp.ndarray:
        mu, std = self.actor(params, key, self.preproc(params, key, convert_jax(obses)))
        return mu, jnp.exp(std)

    def action_discrete(self, obs):
        prob = np.asarray(self._get_actions(self.params, obs))
        return np.expand_dims(
            np.stack([np.random.choice(self.action_size[0], p=p) for p in prob], axis=0),
            axis=1,
        )

    def action_continuous(self, obs):
        mu, std = self._get_actions(self.params, obs)
        return np.random.normal(mu, std)

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jax.nn.softmax(prob)
        prob = jnp.clip(prob, 1e-8, 1.0)
        prob = prob / jnp.sum(prob, axis=-1, keepdims=True)
        action = action.astype(jnp.int32)
        if out_prob:
            return prob, jnp.log(jnp.take_along_axis(prob, action, axis=1))
        else:
            return jnp.log(jnp.take_along_axis(prob, action, axis=1))

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        if out_prob:
            return prob, -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )

    def _loss_continuous(self):
        pass

    def _loss_discrete(self):
        pass

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(self.lossque):.3f}"

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
        if not self.lr_annealing or self.params is None:
            return

        schedule = optax.linear_schedule(
            init_value=self.learning_rate,
            end_value=0.0,
            transition_steps=self._lr_annealing_transition_steps(total_timesteps),
        )
        self.optimizer = self._make_optimizer(schedule)
        self.opt_state = self.optimizer.init(self.params)

    def run_training_loop(self, ctx):
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
        obs, info = self.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.lossque = deque(maxlen=10)
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        eval_result = None
        score = 0.0
        eplen = 0
        original = 0.0
        have_original = False
        for steps in ctx.pbar:
            actions = self.actions(obs)
            step_action = _normalize_action_for_step(self.conv_action(actions))
            next_obs, reward, terminated, truncated, info = self.env.step(step_action)
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.buffer.add(obs, actions, [reward], next_obs, [terminated], [truncated])
            score += float(reward)
            eplen += 1
            step_original = extract_original_reward(info)
            if step_original is not None:
                have_original = True
                original += float(step_original)
            obs = next_obs

            if terminated or truncated:
                lives = extract_lives(info)
                emit_original = have_original and (lives is None or lives == 0)
                self.rollout_tracker.record(
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
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if (steps + 1) % self.batch_size == 0:
                loss = self.train_step(steps, logger_run=ctx.logger_run)
                self.lossque.append(loss)

            if steps % ctx.eval_freq == 0:
                eval_result = self.eval(ctx, steps)

            if steps % ctx.log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                ctx.pbar.set_description(self.description(eval_result))

    def learn_VectorizedEnv(self, ctx):
        self.lossque = deque(maxlen=10)
        self.rollout_tracker = EpisodeTracker(ctx.logger_run.log_metric, ctx.log_interval)
        eval_result = None
        scores = np.zeros([self.worker_size], dtype=np.float64)
        eplens = np.zeros([self.worker_size], dtype=np.int32)
        originals = np.zeros([self.worker_size], dtype=np.float64)
        original_present = np.zeros([self.worker_size], dtype=bool)
        # Workers that ended an episode last step emit an autoreset dummy step
        # (action ignored, reward 0, fresh obs). On-policy rollouts have a fixed
        # per-worker length, so the dummy can't be dropped; instead flag it
        # terminal so it contributes a zero-value target and never bridges the
        # two episodes in the return. prev_done chains off the *real* env dones,
        # and the same mask keeps the dummy out of the rollout episode stats.
        prev_done = None
        convert_action = self.conv_action if self.action_type == "continuous" else None

        def send(actions):
            self.env.step(convert_action(actions) if convert_action else actions)

        # Pipeline the async env between updates. At an update boundary, delay
        # the next send until train_step finishes so the new rollout cannot start
        # with an action sampled from the previous policy.
        obs = self.env.current_obs()
        actions = self.actions([obs])
        send(actions)

        for steps in ctx.pbar:
            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            train_due = (steps + self.worker_size) % (self.batch_size * self.worker_size) == 0
            if not train_due:
                # Keep the async overlap except when train_step changes the policy.
                next_actions = self.actions([next_obses])
                send(next_actions)

            done = np.logical_or(terminateds, truncateds)
            real_reset = vector_real_reset_mask(self.env, terminateds, truncateds, infos)
            autoreset = vector_autoreset_mask(self.env, terminateds, truncateds, infos)
            active = np.ones(self.worker_size, dtype=bool) if prev_done is None else ~prev_done
            scores[active] += rewards[active]
            eplens[active] += 1
            step_original, step_original_present = extract_vector_original_rewards(
                infos, self.worker_size
            )
            active_original = active & step_original_present
            originals[active_original] += step_original[active_original]
            original_present[active_original] = True

            if prev_done is not None and prev_done.any():
                # Flag the dummy step terminal AND zero its reward so it is fully
                # inert (zero-value target, no episode bridge), independent of
                # whatever the env reports on the discarded autoreset step.
                terminateds = np.where(prev_done, True, terminateds)
                rewards = np.where(prev_done, np.float32(0.0), rewards)
            self.buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            for idx in np.where(done & active)[0]:
                emit_original = original_present[idx] and real_reset[idx]
                self.rollout_tracker.record(
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

            if train_due:
                loss = self.train_step(steps, logger_run=ctx.logger_run)
                self.lossque.append(loss)
                next_actions = self.actions([next_obses])
                send(next_actions)

            # Advance the pipeline: the action just sent belongs to next_obses.
            obs = next_obses
            actions = next_actions

            if steps % ctx.eval_freq == 0:
                eval_result = self.eval(ctx, steps)

            if steps % ctx.log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                ctx.pbar.set_description(self.description(eval_result))

    def eval(self, ctx, steps):
        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            self.actions,
            logger_run=ctx.logger_run,
            steps=steps,
            conv_action=self.conv_action,
        )

    def test(self, episode=10):
        with self.logger as logger_run:
            self.test_eval_env(logger_run, episode)

    def test_eval_env(self, logger_run, episode):
        record_test_fn = getattr(self, "record_test_fn", record_and_test)
        return record_test_fn(
            self.env_builder,
            logger_run,
            self.actions,
            episode,
            conv_action=self.conv_action,
        )
