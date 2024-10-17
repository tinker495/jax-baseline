from collections import deque
import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from tqdm.auto import trange

from jax_baselines.common.logger import TensorboardLogger
from jax_baselines.common.cpprb_buffers import EpochBuffer
from jax_baselines.common.utils import convert_jax, key_gen, restore, save, select_optimizer
from jax_baselines.common.env_builer import VectorizedEnv


class Actor_Critic_Policy_Gradient_Family(object):
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
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        self.name = "Actor_Critic_Policy_Gradient_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.log_dir = log_dir
        self.full_tensorboard_log = full_tensorboard_log

        self.params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate)

        self.get_env_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = restore(path)

    def get_memory_setup(self):
        self.buffer = EpochBuffer(
            self.batch_size,
            self.observation_space,
            self.worker_size,
            [1] if self.action_type == "discrete" else self.action_size,
        )

    def get_env_setup(self):
        self.env = self.env_builder(self.num_workers)
        self.eval_env = self.env_builder(1)

        print("----------------------env------------------------")
        if isinstance(self.env, VectorizedEnv):
            print("Vectorized environmet")
            env_info = self.env.env_info
            self.observation_space = [list(env_info["observation_space"].shape)]
            if not isinstance(env_info["action_space"], spaces.Box):
                self.action_size = [env_info["action_space"].n]
                self.action_type = "discrete"
                self.conv_action = lambda a: a[0]
            else:
                self.action_size = [env_info["action_space"].shape[0]]
                self.action_type = "continuous"
                self.conv_action = lambda a: np.clip(a, -3.0, 3.0) / 3.0
            self.worker_size = self.env.worker_num
            self.env_type = "VectorizedEnv"

        elif isinstance(self.env, gym.Env) or isinstance(self.env, gym.Wrapper):
            print("Single environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            if not isinstance(action_space, spaces.Box):
                self.action_size = [action_space.n]
                self.action_type = "discrete"
                self.conv_action = lambda a: a[0]
            else:
                self.action_size = [action_space.shape[0]]
                self.action_type = "continuous"
                self.conv_action = lambda a: np.clip(a, -3.0, 3.0) / 3.0
            self.worker_size = 1
            self.env_type = "SingleEnv"

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

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"

        return discription

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="A2C",
        run_name="A2C"
    ):
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
        for steps in pbar:
            actions = self.actions(obs)[0]
            next_obs, reward, terminated, truncated, info = self.env.step(self.conv_action(actions)[0])
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.buffer.add(obs, actions, [reward], next_obs, [terminated], [truncated])
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def learn_VectorizedEnv(self, pbar, callback=None, log_interval=1000):
        self.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            obs = self.env.current_obs()
            actions = self.actions([obs])
            self.env.step(actions)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            self.buffer.add(
                [obs], actions, rewards, [next_obses], terminateds, truncateds
            )

            if (steps + self.worker_size) % (self.batch_size * self.worker_size) == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

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
                actions = self.actions(obs)[0]
                observation, reward, terminated, truncated, info = self.eval_env.step(self.conv_action(actions))
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
        from gymnasium.wrappers import RecordVideo
        directory = self.logger_run.get_local_path("video")
        os.makedirs(directory, exist_ok=True)

        Render_env = RecordVideo(self.eval_env, directory, episode_trigger=lambda x: True)
        Render_env.start_video_recorder()
        total_rewards = []
        for i in range(episode):
            with Render_env:
                obs, info = Render_env.reset()
                obs = [np.expand_dims(obs, axis=0)]
                terminated = False
                truncated = False
                episode_rew = 0
                eplen = 0
                while not terminated and not truncated:
                    actions = self.actions(obs)[0]
                    observation, reward, terminated, truncated, info = self.eval_env.step(self.conv_action(actions))
                    obs = [np.expand_dims(observation, axis=0)]
                    episode_rew += reward
                    eplen += 1
            print("episod reward :", episode_rew, "episod len :", eplen)
            total_rewards.append(episode_rew)
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"reward : {avg_reward} +- {std_reward}(std)")