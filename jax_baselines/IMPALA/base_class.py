import time
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.env_info import get_worker_env_info
from jax_baselines.core.replay_protocol import (
    WorkerReplayBufferFactory,
    require_replay_factory,
)
from jax_baselines.core.runtime_adapters import make_progress
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.core.serialization import restore, save
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.returns import get_vtrace
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class IMPALA_Family(object):
    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        rho_max=1.0,
        log_interval=1,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory: OptimizerFactory | None = None,
        worker_replay_factory: WorkerReplayBufferFactory | None = None,
    ):
        self.workers = workers
        self.model_builder_maker = model_builder_maker
        self.worker_replay_factory = worker_replay_factory
        self.runtime = runtime
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
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.use_entropy_adv_shaping = use_entropy_adv_shaping
        self.entropy_adv_shaping_kappa = entropy_adv_shaping_kappa
        self.rho_max = rho_max
        self.cut_max = 1.0
        self.log_dir = log_dir

        self.params = None
        self.target_params = None
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

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

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
                + 0.5 * jnp.log(2 * jnp.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * jnp.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )

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
        adv = rewards + self.gamma * (1.0 - terminateds) * vs_t_plus_1 - value
        adv = rho * adv
        return vs, rho, adv

    def get_memory_setup(self):
        self.buffer = self.runtime.create_impala_buffer(
            self.buffer_size,
            self.worker_num,
            self.observation_space,
            discrete=(self.action_type == "discrete"),
            action_space=self.action_size,
            sample_size=self.sample_size,
            seed=self.seed,
        )

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def get_actor_builder(self):
        action_type = self.action_type
        action_size = self.action_size

        def builder():
            if action_type == "discrete":

                def actor(actor_model, preproc, params, obses, key=None):
                    prob = actor_model(params, key, preproc(params, key, convert_jax(obses)))
                    return jax.nn.softmax(prob)

                def get_action_prob(actor, params, obses):
                    prob = np.asarray(actor(params, obses))
                    action = np.random.choice(action_size[0], p=prob[0])
                    return action, np.log(prob[0][action])

                def convert_action(action):
                    return int(action)

            elif action_type == "continuous":

                def actor(actor_model, preproc, params, obses, key=None):
                    mean, log_std = actor_model(
                        params, key, preproc(params, key, convert_jax(obses))
                    )
                    return mean, log_std

                def get_action_prob(actor, params, obses):
                    mean, log_std = actor(params, obses)
                    std = np.exp(log_std)
                    action = np.random.normal(mean, std)
                    return action, -(
                        0.5
                        * np.sum(
                            np.square((action - mean) / (std + 1e-7)),
                            axis=-1,
                            keepdims=True,
                        )
                        + np.sum(log_std, axis=-1, keepdims=True)
                        + 0.5 * np.log(2 * np.pi) * np.asarray(action.shape[-1], dtype=np.float32)
                    )

                def convert_action(action):
                    return np.clip(action[0], -3.0, 3.0) / 3.0

            return actor, get_action_prob, convert_action

        return builder

    def description(self):
        return "loss : {:.3f} |".format(np.mean(self.lossque))

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=1000,
        run_name="IMPALA",
        reset_num_timesteps=True,
        replay_wrapper=None,
        experiment_name="experiment",
        logger_factory=None,
        progress_factory=None,
    ):
        progress_factory = progress_factory or make_progress
        pbar = progress_factory(total_trainstep, miniters=log_interval)

        self.logger_server = self.runtime.create_logger_server(
            self.log_dir, run_name, experiment_name, logger_factory
        )

        if self.env_type == "SingleEnv":
            self.learn_SingleEnv(pbar, callback, log_interval)

        self.save_params(self.logger_server.get_log_dir())

    def learn_SingleEnv(self, pbar, callback, log_interval):
        stop = self.runtime.create_event()
        update = [self.runtime.create_event() for i in range(self.worker_num)]
        stop.clear()
        for u in update:
            u.set()

        cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
        param_server = self.runtime.create_param_server(cpu_param)

        worker_replay_factory = require_replay_factory(
            self.worker_replay_factory, "WorkerReplayBufferFactory"
        )
        jobs = []
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
                print("Stop Training")
                self.runtime.wait(jobs, timeout=300)
                self.runtime.shutdown()
                return

        print("Start Training")
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            if stop.is_set():
                print("Stop Training")
                break
            loss, rho = self.train_step(steps)
            self.lossque.append(loss)
            if steps % log_interval == 0:
                pbar.set_description(self.description())

            if steps % self.update_freq == 0:
                cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
                param_server.update_params(cpu_param)
                for u in update:
                    u.set()
        self.logger_server.last_update()
        self.logger_server.close()
        stop.set()
        self.buffer.clear()
        self.runtime.wait(jobs, timeout=300)
        time.sleep(1)
        self.runtime.shutdown()
