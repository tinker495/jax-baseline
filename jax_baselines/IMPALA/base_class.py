import time
from collections import deque
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.checkpoint_state import ACCheckpointState
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.distributed_runtime import DistributedRuntime, ImpalaRolloutNeed
from jax_baselines.core.env_info import get_worker_env_info
from jax_baselines.core.hparams import get_hyper_params
from jax_baselines.core.replay_protocol import (
    WorkerReplayBufferFactory,
    require_replay_factory,
)
from jax_baselines.core.runtime_adapters import make_progress
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.returns import get_vtrace
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class IMPALA_Family:
    critic: Callable[..., Any]
    _train_step: Callable[..., tuple]
    _run_name = "IMPALA"
    _learn_log_interval = 1000

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime: DistributedRuntime,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        rho_max=1.0,
        log_interval=100,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory: OptimizerFactory | None = None,
        worker_replay_factory: WorkerReplayBufferFactory | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ):
        if use_entropy_adv_shaping and (not np.isfinite(ent_coef) or ent_coef < 0):
            raise ValueError("entropy shaping requires finite ent_coef >= 0")
        if use_entropy_adv_shaping and (
            not np.isfinite(entropy_adv_shaping_kappa) or entropy_adv_shaping_kappa <= 1
        ):
            raise ValueError("entropy shaping requires finite entropy_adv_shaping_kappa > 1")
        self.workers = workers
        self.model_builder_maker = model_builder_maker
        self.worker_replay_factory = worker_replay_factory
        self.runtime = runtime
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)
        self.buffer_size = buffer_size
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        set_global_seeds(self.seed)
        self.key_seq = key_gen(self.seed)
        self.update_freq = update_freq

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda
        self.ent_coef = ent_coef
        self.use_entropy_adv_shaping = use_entropy_adv_shaping
        self.entropy_adv_shaping_kappa = entropy_adv_shaping_kappa
        self.rho_max = rho_max
        self.cut_max = 1.0
        self.log_dir = log_dir

        self.actor_params = None
        self.critic_params = None
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.optimizer = self._make_optimizer(self.learning_rate)
        self.model_builder = None
        self.actor_builder = None

        self.get_env_setup()
        self.get_memory_setup()

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()
        # Update PRNG carry; drawn after model init so initial params are unchanged.
        self._train_key = next(self.key_seq)

    def save_params(self, path):
        self.checkpoint_store.save(
            path,
            ACCheckpointState(actor_params=self.actor_params, critic_params=self.critic_params),
        )

    def load_params(self, path):
        state = self.checkpoint_store.restore(path)
        if not isinstance(state, ACCheckpointState):
            raise TypeError("Expected ACCheckpointState with separate actor and critic parameters")
        self.actor_params = jax.device_put(state.actor_params)
        self.critic_params = jax.device_put(state.critic_params)

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

    def _critic_loss(self, critic_params, actor_params, obses, targets, key):
        return jnp.mean(jnp.square(targets - self.critic(critic_params, actor_params, key, obses)))

    def get_env_setup(self):
        (
            self.observation_space,
            self.action_size,
            self.env_type,
            self.action_type,
        ) = get_worker_env_info(self.workers, self.runtime.worker_info, include_action_type=True)
        self.worker_num = len(self.workers)
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", len(self.workers))
        print("-------------------------------------------------")

        self.get_logprob = (
            self.get_logprob_discrete
            if self.action_type == "discrete"
            else self.get_logprob_continuous
        )

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jnp.clip(jax.nn.softmax(prob), 1e-5, 1.0)
        action = action.astype(jnp.int32)
        log_prob = jnp.log(jnp.take_along_axis(prob, action, axis=1))
        return (prob, log_prob) if out_prob else log_prob

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        log_prob = -(
            0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
            + jnp.sum(log_std, axis=-1, keepdims=True)
            + 0.5 * jnp.log(2 * jnp.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
        )
        return (prob, log_prob) if out_prob else log_prob

    def _compute_vtrace(
        self, pi_prob, mu_log_prob, rewards, terminateds, truncateds, value, next_value
    ):
        """V-trace targets and clipped-IS advantages shared by every IMPALA
        variant (A2C/PPO/TPPO/SPO). Inputs are per-worker stacked sequences."""
        rho_raw = jnp.exp(pi_prob - mu_log_prob)
        rho = jnp.minimum(rho_raw, self.rho_max)
        c_t = self.lamda * jnp.minimum(rho, self.cut_max)
        vs = jax.vmap(get_vtrace, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            rewards, rho, c_t, terminateds, truncateds, value, next_value, self.gamma
        )
        vs_t_plus_1 = jax.vmap(
            lambda v, nv, t: jnp.where(
                t == 1, nv, jnp.concatenate([v[1:], jnp.expand_dims(nv[-1], axis=-1)])
            ),
            in_axes=(0, 0, 0),
        )(vs, next_value, truncateds)
        bootstrap = jnp.where(terminateds.astype(bool), 0.0, vs_t_plus_1)
        adv = rewards + self.gamma * bootstrap - value
        adv = rho * adv
        return vs, rho, adv

    def get_memory_setup(self):
        self.buffer = self.runtime.create_impala_buffer(
            ImpalaRolloutNeed(
                replay_size=self.buffer_size,
                actor_num=self.worker_num,
                observation_space=self.observation_space,
                discrete=(self.action_type == "discrete"),
                action_space=self.action_size,
                sample_size=self.sample_size,
                seed=self.seed,
            )
        )

    def setup_model(self):
        pass

    def train_step(self, steps):
        (
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            self._train_key,
            critic_loss,
            actor_loss,
            entropy_loss,
            rho,
            targets,
        ) = self._train_step(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            self._train_key,
            *jax.device_put(self.buffer.sample()),
        )

        if steps % self.log_interval == 0:
            log_dict = {
                "loss/critic_loss": critic_loss,
                "loss/actor_loss": actor_loss,
                "loss/entropy_loss": entropy_loss,
                "loss/mean_rho": rho,
                "loss/mean_target": targets,
            }
            self.logger_server.log_trainer(
                steps, {key: float(value) for key, value in jax.device_get(log_dict).items()}
            )
        return critic_loss

    def get_actor_builder(self):
        """Return a worker-side factory for the compiled behavior-action stage.

        ``sample_action(actor_model, actor_params, obses, key)`` runs inference,
        sampling and the behavior log-prob in one call and returns the advanced key.
        """
        action_type = self.action_type

        def builder():
            if action_type == "discrete":

                def sample_action(actor_model, actor_params, obses, key):
                    key, sample_key = jax.random.split(key)
                    prob = jax.nn.softmax(
                        actor_model(actor_params, None, convert_normalized_obs(obses))
                    )[0]
                    action = jax.random.categorical(sample_key, jnp.log(prob))
                    return action, jnp.log(prob[action]), key

                def convert_action(action):
                    return int(action)

            elif action_type == "continuous":

                def sample_action(actor_model, actor_params, obses, key):
                    key, sample_key = jax.random.split(key)
                    mean, log_std = actor_model(actor_params, None, convert_normalized_obs(obses))
                    std = jnp.exp(log_std)
                    action = mean + std * jax.random.normal(sample_key, mean.shape, mean.dtype)
                    return (
                        action,
                        -(
                            0.5
                            * jnp.sum(
                                jnp.square((action - mean) / (std + 1e-7)),
                                axis=-1,
                                keepdims=True,
                            )
                            + jnp.sum(log_std, axis=-1, keepdims=True)
                            + 0.5
                            * jnp.log(2 * jnp.pi)
                            * jnp.asarray(action.shape[-1], dtype=jnp.float32)
                        ),
                        key,
                    )

                def convert_action(action):
                    return action[0]

            return sample_action, convert_action

        return builder

    def run_name_update(self, run_name):
        return run_name

    def _optimizer_updates_per_train_step(self):
        return 1

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=None,
        run_name=None,
        reset_num_timesteps=True,
        replay_wrapper=None,
        experiment_name="experiment",
        logger_factory=None,
        progress_factory=None,
    ):
        if log_interval is None:
            log_interval = self._learn_log_interval
        if run_name is None:
            run_name = self._run_name
        run_name = self.run_name_update(run_name)
        progress_factory = progress_factory or make_progress
        pbar = progress_factory(total_trainstep, miniters=log_interval)

        logger_server = None
        try:
            logger_server = self.runtime.create_logger_server(
                self.log_dir, run_name, experiment_name, logger_factory
            )
            self.logger_server = logger_server
            logger_server.register_hparams(get_hyper_params(self))

            if self.env_type == "SingleEnv":
                self.learn_SingleEnv(pbar, callback, log_interval)

            self.save_params(logger_server.get_log_dir())
        finally:
            try:
                if logger_server is not None:
                    try:
                        logger_server.last_update()
                    finally:
                        logger_server.close()
            finally:
                self.runtime.shutdown()

    def learn_SingleEnv(self, pbar, callback, log_interval):
        started_at = time.perf_counter()
        completed_iterations = 0
        update_steps = 0
        stop = self.runtime.create_event()
        stop.clear()
        jobs = []
        try:
            update = [self.runtime.create_event() for _ in range(self.worker_num)]
            for u in update:
                u.set()

            # Workers receive host copies: one explicit download per broadcast.
            cpu_param = jax.device_get(self.actor_params)
            param_server = self.runtime.create_param_server(cpu_param)

            worker_replay_factory = require_replay_factory(
                self.worker_replay_factory, "WorkerReplayBufferFactory"
            )
            for idx in range(self.worker_num):
                jobs.append(
                    self.workers[idx].run(
                        self.batch_size,
                        self.buffer.queue_info(),
                        worker_replay_factory,
                        self.model_builder,
                        self.actor_builder,
                        param_server,
                        update[idx],
                        self.logger_server,
                        stop,
                        seed=self.seed + idx,
                    )
                )

            print("Start Warmup")
            while self.buffer.queue_is_empty():
                time.sleep(1)
                if stop.is_set():
                    raise RuntimeError("distributed worker stopped during warmup")

            print("Start Training")
            self.lossque = deque(maxlen=10)
            for steps in pbar:
                if stop.is_set():
                    raise RuntimeError("distributed worker stopped during training")
                self.lossque.append(self.train_step(steps))
                completed_iterations = steps + 1
                update_steps += self._optimizer_updates_per_train_step()
                if steps % log_interval == 0:
                    self.logger_server.log_trainer(
                        steps,
                        {
                            "progress/update_steps": update_steps,
                            "time/elapsed_seconds": time.perf_counter() - started_at,
                        },
                    )
                    pbar.set_description(
                        f"loss : {np.mean(jax.device_get(tuple(self.lossque))):.3f} |"
                    )

                if steps % self.update_freq == 0:
                    cpu_param = jax.device_get(self.actor_params)
                    param_server.update_params(cpu_param)
                    for u in update:
                        u.set()
        finally:
            stop.set()
            try:
                self.buffer.clear()
            finally:
                self.runtime.wait(jobs, timeout=300)
                jax.block_until_ready((self.actor_params, self.critic_params))
                self.logger_server.log_trainer(
                    completed_iterations,
                    {
                        "progress/update_steps": update_steps,
                        "time/elapsed_seconds": time.perf_counter() - started_at,
                    },
                )
