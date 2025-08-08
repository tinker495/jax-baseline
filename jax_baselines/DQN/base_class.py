import os
from collections import deque

import gymnasium as gym
import numpy as np
from tqdm.auto import trange

from jax_baselines.common.cpprb_buffers import (
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from jax_baselines.common.env_builder import VectorizedEnv
from jax_baselines.common.logger import TensorboardLogger
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.schedules import ConstantSchedule, LinearSchedule
from jax_baselines.common.utils import key_gen, restore, save


class Q_Network_Family(object):
    def __init__(
        self,
        env_builder: callable,
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
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
        compress_memory=False,
    ):
        self.name = "Q_Network_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
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
        self.full_tensorboard_log = full_tensorboard_log
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
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-3 / self.batch_size)

        self.compress_memory = compress_memory

        self.get_env_setup()
        self.get_memory_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        self.env = self.env_builder(self.num_workers)
        self.eval_env = self.env_builder(1)

        print("----------------------env------------------------")
        if isinstance(self.env, VectorizedEnv):
            print("Vectorized environmet")
            env_info = self.env.env_info
            self.observation_space = [list(env_info["observation_space"].shape)]
            self.action_size = [env_info["action_space"].n]
            self.worker_size = self.env.worker_num
            self.env_type = "VectorizedEnv"

        elif isinstance(self.env, gym.Env) or isinstance(self.env, gym.Wrapper):
            print("Single environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.n]
            self.worker_size = 1
            self.env_type = "SingleEnv"

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")

    def get_memory_setup(self):
        if self.prioritized_replay:
            if self.n_step_method:
                self.replay_buffer = PrioritizedNstepReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    1,
                    self.worker_size,
                    self.n_step,
                    self.gamma,
                    self.prioritized_replay_alpha,
                    False,
                    self.prioritized_replay_eps,
                )
            else:
                self.replay_buffer = PrioritizedReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.prioritized_replay_alpha,
                    1,
                    False,
                    self.prioritized_replay_eps,
                )

        else:
            if self.n_step_method:
                self.replay_buffer = NstepReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    1,
                    self.worker_size,
                    self.n_step,
                    self.gamma,
                )
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space, 1)

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def actions(self, obs, epsilon):
        if epsilon <= np.random.uniform(0, 1):
            actions = np.asarray(
                self._get_actions(
                    self.params, obs, next(self.key_seq) if self.param_noise else None
                )
            )
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size, 1])
        return actions

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"

        if self.param_noise:
            return discription
        else:
            return discription + f", epsilon : {self.update_eps:.3f}"

    def run_name_update(self, run_name):
        if self.munchausen:
            run_name = "M-" + run_name
        if (
            self.param_noise
            & self.dueling_model
            & self.double_q
            & self.n_step_method
            & self.prioritized_replay
        ):
            run_name = f"Rainbow({self.n_step} step)_" + run_name
        else:
            if self.param_noise:
                run_name = "Noisy_" + run_name
            if self.dueling_model:
                run_name = "Dueling_" + run_name
            if self.double_q:
                run_name = "Double_" + run_name
            if self.n_step_method:
                run_name = "{}Step_".format(self.n_step) + run_name
            if self.prioritized_replay:
                run_name = run_name + "+PER"
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="Q_network",
        run_name="Q_network",
    ):
        run_name = self.run_name_update(run_name)

        if self.param_noise:
            self.exploration = ConstantSchedule(0)
        else:
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )
        self.update_eps = 1.0
        self.eval_freq = ((total_timesteps // 100) // self.worker_size) * self.worker_size

        pbar = trange(0, total_timesteps, self.worker_size, miniters=log_interval)
        self.logger = TensorboardLogger(run_name, experiment_name, self.log_dir, self)
        with self.logger as self.logger_run:
            if self.env_type == "SingleEnv":
                self.learn_SingleEnv(pbar, callback, log_interval)
            if self.env_type == "VectorizedEnv":
                self.learn_VectorizedEnv(pbar, callback, log_interval)

            self.eval(total_timesteps)

            self.save_params(self.logger_run.get_local_path("params"))

    def learn_SingleEnv(self, pbar, callback=None, log_interval=1000):
        obs, info = self.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            actions = self.actions(obs, self.update_eps)
            next_obs, reward, terminated, truncated, info = self.env.step(actions[0][0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.replay_buffer.add(obs, actions[0], reward, next_obs, terminated, truncated)
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if steps > self.learning_starts and steps % self.train_freq == 0:
                self.update_eps = self.exploration.value(steps)
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def learn_VectorizedEnv(self, pbar, callback=None, log_interval=1000):
        self.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            self.update_eps = self.exploration.value(steps)
            obs = self.env.current_obs()
            actions = self.actions([obs], self.update_eps)
            self.env.step(actions)

            if steps > self.learning_starts and steps % self.train_freq == 0:
                for idx in range(self.worker_size):
                    loss = self.train_step(steps + idx, self.gradient_steps)
                    self.lossque.append(loss)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            self.replay_buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def eval(self, steps):
        original_rewards = []
        total_reward = np.zeros(self.eval_eps)
        total_ep_len = np.zeros(self.eval_eps)
        total_truncated = np.zeros(self.eval_eps)

        obs, info = self.eval_env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        have_original_reward = "original_reward" in info.keys()
        have_lives = "lives" in info.keys()
        if have_original_reward:
            original_reward = info["original_reward"]
        terminated = False
        truncated = False
        eplen = 0

        for ep in range(self.eval_eps):
            while not terminated and not truncated:
                actions = self.actions(obs, 0.001)
                observation, reward, terminated, truncated, info = self.eval_env.step(actions[0][0])
                obs = [np.expand_dims(observation, axis=0)]
                if have_original_reward:
                    original_reward += info["original_reward"]
                total_reward[ep] += reward
                eplen += 1

            total_ep_len[ep] = eplen
            total_truncated[ep] = float(truncated)
            if have_original_reward:
                if have_lives:
                    if info["lives"] == 0:
                        original_rewards.append(original_reward)
                        original_reward = 0
                else:
                    original_rewards.append(original_reward)
                    original_reward = 0

            obs, info = self.eval_env.reset()
            obs = [np.expand_dims(obs, axis=0)]
            terminated = False
            truncated = False
            eplen = 0

        if have_original_reward:
            mean_original_score = np.mean(original_rewards)
        mean_reward = np.mean(total_reward)
        mean_ep_len = np.mean(total_ep_len)

        if self.logger_run:
            if have_original_reward:
                self.logger_run.log_metric("env/original_reward", mean_original_score, steps)
            self.logger_run.log_metric("env/episode_reward", mean_reward, steps)
            self.logger_run.log_metric("env/episode len", mean_ep_len, steps)
            self.logger_run.log_metric("env/time over", np.mean(total_truncated), steps)

        if have_original_reward:
            eval_result = {
                "mean_reward": mean_reward,
                "mean_ep_len": mean_ep_len,
                "mean_original_score": mean_original_score,
            }
        else:
            eval_result = {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}
        return eval_result

    def test(self, episode=10):
        with self.logger as self.logger_run:
            self.test_eval_env(episode)

    def test_eval_env(self, episode):
        from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

        directory = self.logger_run.get_local_path("video")
        os.makedirs(directory, exist_ok=True)

        test_env = self.env_builder(1, render_mode="rgb_array")
        Render_env = RecordVideo(test_env, directory, episode_trigger=lambda x: True)
        Render_env = RecordEpisodeStatistics(Render_env)
        total_rewards = []
        with Render_env:
            for i in range(episode):
                obs, info = Render_env.reset()
                obs = [np.expand_dims(obs, axis=0)]
                terminated = False
                truncated = False
                episode_rew = 0
                eplen = 0
                while not terminated and not truncated:
                    actions = self.actions(obs, 0.001)
                    observation, reward, terminated, truncated, info = Render_env.step(
                        actions[0][0]
                    )
                    obs = [np.expand_dims(observation, axis=0)]
                    episode_rew += reward
                    eplen += 1
                print("episod reward :", episode_rew, "episod len :", eplen)
                total_rewards.append(episode_rew)
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"reward : {avg_reward} +- {std_reward}(std)")
