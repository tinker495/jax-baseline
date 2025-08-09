import time
from collections import deque

import jax
import numpy as np
import ray
from tqdm.auto import trange

from jax_baselines.APE_X.common_servers import Logger_server, Param_server
from jax_baselines.common.env_info import get_remote_env_info
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.replay_factory import make_multi_prioritized_buffer
from jax_baselines.common.utils import get_hyper_params, key_gen, restore, save


class Ape_X_Deteministic_Policy_Gradient_Family(object):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        exploration_initial_eps=0.9,
        exploration_decay=0.7,
        batch_num=16,
        mini_batch_size=512,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        gradient_steps=1,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        scaled_by_reset=False,
        simba=False,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
        param_broadcast_freq=20,
    ):
        self.workers = workers
        self.model_builder_maker = model_builder_maker
        self.m = manager
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.learning_starts = learning_starts
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_num = batch_num
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_num * mini_batch_size
        self.target_network_update_tau = target_network_update_tau
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
        self.full_tensorboard_log = full_tensorboard_log
        self.n_step_method = n_step > 1
        self.n_step = n_step
        self.munchausen_alpha = 0.9
        self.munchausen_entropy_tau = 0.03

        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-3 / self.batch_size)
        self.model_builder = None
        self.actor_builder = None

        self.compress_memory = compress_memory
        self.param_broadcast_freq = param_broadcast_freq

        self.get_env_setup()
        self.get_memory_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        self.observation_space, self.action_size, self.env_type = get_remote_env_info(self.workers)
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", len(self.workers))
        print("-------------------------------------------------")

    def get_memory_setup(self):
        self.replay_buffer = make_multi_prioritized_buffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            alpha=self.prioritized_replay_alpha,
            action_shape_or_n=self.action_size,
            n_step=self.n_step,
            gamma=self.gamma,
            manager=self.m,
            compress_memory=self.compress_memory,
            eps=self.prioritized_replay_eps,
        )

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def discription(self):
        return "buffer len : {} loss : {:.3f} |".format(
            len(self.replay_buffer), np.mean(self.lossque)
        )

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=1000,
        run_name="APE_X_DPG",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        if self.n_step_method:
            run_name = "{}Step_".format(self.n_step) + run_name
        self.update_eps = 1.0

        pbar = trange(total_trainstep, miniters=log_interval)

        self.logger_server = Logger_server.remote(self.log_dir, run_name)
        try:
            hparams = get_hyper_params(self)
            # send serializable dict to the logger actor
            self.logger_server.register_hparams.remote(hparams)
        except Exception:
            pass

        if self.env_type == "unity":
            self.learn_unity(pbar, callback, log_interval)
        if self.env_type == "SingleEnv":
            self.learn_SingleEnv(pbar, callback, log_interval)

        self.save_params(ray.get(self.logger_server.get_log_dir.remote()))

    def learn_unity(self, pbar, callback, log_interval):
        pass

    def learn_SingleEnv(self, pbar, callback, log_interval):
        stop = self.m.Event()
        worker_num = len(self.workers)
        update = [self.m.Event() for i in range(worker_num)]
        stop.clear()
        for u in update:
            u.clear()

        cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
        param_server = Param_server.remote(cpu_param)

        self.logger_server.add_multiline.remote(
            [
                self.exploration_initial_eps
                ** (1 + self.exploration_decay * idx / (worker_num - 1))
                for idx in range(worker_num)
            ]
        )
        jobs = []
        for idx in range(worker_num):
            eps = self.exploration_initial_eps ** (
                1 + self.exploration_decay * idx / (worker_num - 1)
            )
            jobs.append(
                self.workers[idx].run.remote(
                    2000,
                    self.replay_buffer.buffer_info(),
                    self.model_builder,
                    self.actor_builder,
                    param_server,
                    self.logger_server,
                    update[idx],
                    stop,
                    eps,
                )
            )
            time.sleep(0.1)

        print("Start Warmup")
        while len(self.replay_buffer) < self.learning_starts:
            time.sleep(1)
            if stop.is_set():
                print("Stop Training")
                _, still_running = ray.wait(jobs, timeout=300)
                self.m.shutdown()
                return

        print("Start Training")
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            if stop.is_set():
                print("Stop Training")
                break
            loss = self.train_step(steps, self.gradient_steps)
            self.lossque.append(loss)
            if steps % log_interval == 0:
                pbar.set_description(self.discription())
            if steps % self.param_broadcast_freq == 0:
                cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
                param_server.update_params.remote(cpu_param)
                for u in update:
                    u.set()
        self.logger_server.last_update.remote()
        stop.set()
        _, still_running = ray.wait(jobs, timeout=300)
        time.sleep(1)
        self.m.shutdown()


# Param_server and Logger_server are provided by `jax_baselines.APE_X.common_servers`
