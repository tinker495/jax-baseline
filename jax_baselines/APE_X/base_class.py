import time
from collections import deque

import jax
import numpy as np

from jax_baselines.APE_X.exploration import worker_epsilons
from jax_baselines.core.checkpoint_store import (
    CheckpointStore,
    checkpoint_store_or_default,
)
from jax_baselines.core.distributed_runtime import DistributedRuntime
from jax_baselines.core.env_info import get_worker_env_info
from jax_baselines.core.hparams import get_hyper_params
from jax_baselines.core.replay_protocol import (
    ApeXReplayFactory,
    PriorityNeed,
    SharedPrioritizedReplayNeed,
    require_replay_factory,
)
from jax_baselines.core.runtime_adapters import make_progress
from jax_baselines.core.seeding import key_gen, set_global_seeds
from jax_baselines.optim import OptimizerFactory, require_optimizer_factory


class Ape_X_Family(object):
    _run_name = "APE_X"

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime: DistributedRuntime,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.9,
        exploration_decay=0.7,
        batch_num=16,
        mini_batch_size=512,
        double_q=False,
        dueling_model=False,
        n_step=1,
        learning_starts=1000,
        target_network_update_freq=2000,
        gradient_steps=1,
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
        apex_replay_factory: ApeXReplayFactory | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ):
        self.workers = workers
        self.model_builder_maker = model_builder_maker
        self.runtime = runtime
        self.apex_replay_factory = apex_replay_factory
        self.checkpoint_store = checkpoint_store_or_default(checkpoint_store)
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        set_global_seeds(self.seed)
        self.key_seq = key_gen(self.seed)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_num = batch_num
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_num * mini_batch_size
        self.target_network_update_freq = int(target_network_update_freq)
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_decay = exploration_decay
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps
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

        self.params = None
        self.target_params = None
        self.train_steps_count = 0
        self.optimizer_factory = require_optimizer_factory(optimizer_factory)
        self.optimizer = self._make_optimizer(self.learning_rate)
        self.model_builder = None
        self.actor_builder = None

        self.compress_memory = compress_memory

        self.get_env_setup()
        self.get_memory_setup()

        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()

    def save_params(self, path):
        self.checkpoint_store.save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = self.checkpoint_store.restore(path)

    def _make_optimizer(self, learning_rate):
        return self.optimizer_factory(learning_rate)

    def get_env_setup(self):
        self.observation_space, self.action_size, self.env_type = get_worker_env_info(
            self.workers, self.runtime.worker_info
        )
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", len(self.workers))
        print("-------------------------------------------------")

    def get_memory_setup(self):
        replay_topology = require_replay_factory(self.apex_replay_factory, "ApeXReplayFactory")(
            SharedPrioritizedReplayNeed(
                buffer_size=self.buffer_size,
                observation_space=self.observation_space,
                priority=PriorityNeed(
                    alpha=self.prioritized_replay_alpha,
                    eps=self.prioritized_replay_eps,
                ),
                action_shape_or_n=1,
                n_step=self.n_step,
                gamma=self.gamma,
                manager=self.runtime.replay_manager(),
                compress_observations=self.compress_memory,
            )
        )
        self.replay_buffer = replay_topology.shared_buffer
        self.worker_replay_factory = replay_topology.worker_factory

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def description(self):
        return "buffer len : {} loss : {:.3f} |".format(
            len(self.replay_buffer), np.mean(self.lossque)
        )

    def train_step(self, steps, gradient_steps):
        for _ in range(gradient_steps):
            self.train_steps_count += 1
            data = self.replay_buffer.sample(self.batch_size, self.prioritized_replay_beta0)

            (
                self.params,
                self.target_params,
                self.opt_state,
                loss,
                t_mean,
                new_priorities,
            ) = self._invoke_train_step(steps, data)

            self.replay_buffer.update_priorities(data["indexes"], new_priorities)

        if steps % self.log_interval == 0:
            log_dict = {"loss/qloss": float(loss), "loss/targets": float(t_mean)}
            self.logger_server.log_trainer(steps, log_dict)

        return loss

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=1000,
        run_name=None,
        reset_num_timesteps=True,
        replay_wrapper=None,
        experiment_name="experiment",
        logger_factory=None,
        progress_factory=None,
    ):
        if run_name is None:
            run_name = self._run_name
        if self.munchausen:
            run_name = "M-" + run_name
        if self.param_noise:
            run_name = "Noisy_" + run_name
        if self.dueling_model:
            run_name = "Dueling_" + run_name
        if self.double_q:
            run_name = "Double_" + run_name
        if self.n_step_method:
            run_name = "{}Step_".format(self.n_step) + run_name

        progress_factory = progress_factory or make_progress
        pbar = progress_factory(total_trainstep, miniters=log_interval)

        logger_server = None
        try:
            logger_server = self.runtime.create_logger_server(
                self.log_dir, run_name, experiment_name, logger_factory
            )
            self.logger_server = logger_server
            hparams = get_hyper_params(self)
            logger_server.register_hparams(hparams)

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

    def learn_SingleEnv(self, pbar, callback=None, log_interval=1000):
        stop = self.runtime.create_event()
        stop.clear()
        jobs = []
        try:
            worker_num = len(self.workers)
            update = [self.runtime.create_event() for _ in range(worker_num)]
            for u in update:
                u.clear()

            cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
            param_server = self.runtime.create_param_server(cpu_param)

            epsilons = worker_epsilons(
                self.exploration_initial_eps, self.exploration_decay, worker_num
            )
            self.logger_server.add_multiline(epsilons)
            for idx in range(worker_num):
                if self.param_noise:
                    eps = None
                else:
                    eps = epsilons[idx]
                jobs.append(
                    self.workers[idx].run(
                        1000,
                        self.replay_buffer.buffer_info(),
                        self.worker_replay_factory,
                        self.model_builder,
                        self.actor_builder,
                        param_server,
                        self.logger_server,
                        update[idx],
                        stop,
                        eps=eps,
                        seed=self.seed + idx,
                    )
                )
                time.sleep(0.1)

            print("Start Warmup")
            while len(self.replay_buffer) < self.learning_starts:
                time.sleep(1)
                if stop.is_set():
                    raise RuntimeError("distributed worker stopped during warmup")

            print("Start Training")
            self.lossque = deque(maxlen=10)
            for steps in pbar:
                if stop.is_set():
                    raise RuntimeError("distributed worker stopped during training")
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)
                if steps % log_interval == 0:
                    pbar.set_description(self.description())
                if steps % self.target_network_update_freq == 0:
                    cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
                    param_server.update_params(cpu_param)
                    for u in update:
                        u.set()
        finally:
            stop.set()
            self.runtime.wait(jobs, timeout=300)
