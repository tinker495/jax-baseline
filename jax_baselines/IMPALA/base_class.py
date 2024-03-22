import multiprocessing as mp
import time
from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import ray
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from tqdm.auto import trange

from jax_baselines.common.base_classes import (
    TensorboardWriter,
    restore,
    save,
    select_optimizer,
)
from jax_baselines.common.utils import convert_jax, key_gen
from jax_baselines.IMPALA.cpprb_buffers import ImpalaBuffer


class IMPALA_Family(object):
    def __init__(
        self,
        workers,
        model_builder_maker,
        manager=None,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        rho_max=1.0,
        log_interval=1,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        self.name = "IMPALA_Family"
        self.workers = workers
        self.model_builder_maker = model_builder_maker
        self.m = manager if manager is not None else mp.Manager()
        self.buffer_size = buffer_size
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)
        self.update_freq = update_freq

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.rho_max = rho_max
        self.cut_max = 1.0
        self.tensorboard_log = tensorboard_log

        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-2 / self.batch_size)
        self.model_builder = None
        self.actor_builder = None

        self.get_env_setup()
        self.get_memory_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        print("----------------------env------------------------")
        if isinstance(self.workers, UnityEnvironment):
            pass

        elif isinstance(self.workers, list) or isinstance(self.env, gym.Wrapper):
            print("openai gym environmet")
            self.worker_num = len(self.workers)
            env_dict = ray.get(self.workers[0].get_info.remote())
            self.observation_space = [list(env_dict["observation_space"].shape)]
            if not isinstance(env_dict["action_space"], spaces.Box):
                self.action_size = [env_dict["action_space"].n]
                self.action_type = "discrete"
            else:
                self.action_size = [env_dict["action_space"].shape[0]]
                self.action_type = "continuous"
            self.env_type = "gym"

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", len(self.workers))
        print("-------------------------------------------------")

        self.get_logprob = (
            self.get_logprob_discrete
            if self.action_type == "discrete"
            else self.get_logprob_continuous
        )

    def model_builder(self):
        pass

    def actor_builder(self):
        pass

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

    def get_memory_setup(self):
        self.buffer = ImpalaBuffer(
            self.buffer_size,
            self.worker_num,
            self.observation_space,
            discrete=(self.action_type == "discrete"),
            action_space=self.action_size,
            sample_size=self.sample_size,
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

    def discription(self):
        return "loss : {:.3f} |".format(np.mean(self.lossque))

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=1000,
        tb_log_name="IMPALA",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        pbar = trange(total_trainstep, miniters=log_interval)

        self.logger_server = Logger_server.remote(self.tensorboard_log, tb_log_name)

        if self.env_type == "unity":
            self.learn_unity(pbar, callback, log_interval)
        if self.env_type == "gym":
            self.learn_gym(pbar, callback, log_interval)

        # add_hparams(self, self.logger_server, ["score", "loss"])
        self.save_params(ray.get(self.logger_server.get_log_dir.remote()))

    def learn_unity(self, pbar, callback, log_interval):
        pass

    def learn_gym(self, pbar, callback, log_interval):
        stop = self.m.Event()
        update = [self.m.Event() for i in range(self.worker_num)]
        stop.clear()
        for u in update:
            u.set()

        cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
        param_server = Param_server.remote(ray.put(cpu_param))

        jobs = []
        for idx in range(self.worker_num):
            jobs.append(
                self.workers[idx].run.remote(
                    self.batch_size,
                    self.buffer.queue_info(),
                    self.model_builder,
                    self.actor_builder,
                    param_server,
                    update[idx],
                    self.logger_server,
                    stop,
                )
            )

        print("Start Warmup")
        while self.buffer.queue_is_empty():
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
            loss, rho = self.train_step(steps)
            self.lossque.append(loss)
            if steps % log_interval == 0:
                pbar.set_description(self.discription())

            if steps % self.update_freq == 0:
                cpu_param = jax.device_put(self.params, jax.devices("cpu")[0])
                param_server.update_params.remote(ray.put(cpu_param))
                for u in update:
                    u.set()
        self.logger_server.last_update.remote()
        stop.set()
        while not self.buffer.queue.empty():
            self.buffer.queue.get()
        _, still_running = ray.wait(jobs, timeout=300)
        time.sleep(1)
        self.m.shutdown()


@ray.remote
class Param_server(object):
    def __init__(self, params) -> None:
        self.params = params

    def get_params(self):
        return self.params

    def update_params(self, params):
        self.params = params


@ray.remote
class Logger_server(object):
    def __init__(self, log_dir, log_name) -> None:
        self.writer = TensorboardWriter(log_dir, log_name)
        self.step = 0
        self.old_step = 0
        self.save_dict = dict()
        with self.writer as (summary, save_path):
            self.save_path = save_path

    def get_log_dir(self):
        return self.save_path

    def add_multiline(self, eps):
        with self.writer as (summary, _):
            layout = {
                "env": {
                    "episode_reward": [
                        "Multiline",
                        ["env/episode_reward"],
                    ],
                    "original_reward": [
                        "Multiline",
                        ["env/original_reward"],
                    ],
                    "episode_len": [
                        "Multiline",
                        ["env/episode_len"],
                    ],
                    "time_over": [
                        "Multiline",
                        ["env/time_over"],
                    ],
                },
            }
            summary.add_custom_scalars(layout)

    def log_trainer(self, step, log_dict):
        self.step = step
        with self.writer as (summary, _):
            for key, value in log_dict.items():
                summary.add_scalar(key, value, self.step)

    def log_worker(self, log_dict, episode):
        if self.old_step != self.step:
            with self.writer as (summary, _):
                for key, value in self.save_dict.items():
                    summary.add_scalar(key, np.mean(value), self.step)
                self.save_dict = dict()
                self.old_step = self.step
        for key, value in log_dict.items():
            if key in self.save_dict:
                self.save_dict[key].append(value)
            else:
                self.save_dict[key] = [value]

    def last_update(self):
        with self.writer as (summary, _):
            for key, value in self.save_dict.items():
                summary.add_scalar(key, np.mean(value), self.step)
