from collections import deque

import gymnasium as gym
import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from tqdm.auto import trange

from jax_baselines.common.base_classes import (
    TensorboardWriter,
    restore,
    save,
    select_optimizer,
)
from jax_baselines.common.cpprb_buffers import (
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from jax_baselines.common.utils import add_hparams, convert_states, key_gen
from jax_baselines.common.env_builer import Multiworker


class Deteministic_Policy_Gradient_Family(object):
    def __init__(
        self,
        env_builder : callable,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=5e-5,
        buffer_size=50000,
        train_freq=1,
        gradient_steps=1,
        batch_size=32,
        n_step=1,
        learning_starts=1000,
        target_network_update_tau=5e-4,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-3,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        self.name = "Deteministic_Policy_Gradient_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.train_steps_count = 0
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
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
        self.full_tensorboard_log = full_tensorboard_log
        self.n_step_method = n_step > 1
        self.n_step = n_step

        self.params = None
        self.target_params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate, 1e-2 / self.batch_size)

        self.get_env_setup()
        self.get_memory_setup()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = self.target_params = restore(path)

    def get_env_setup(self):
        self.env = self.env_builder(self.num_workers)
        self.eval_env = self.env_builder(1)

        if isinstance(self.env, Multiworker):
            print("multiworker environmet")
            env_info = self.env.env_info
            self.observation_space = [list(env_info["observation_space"].shape)]
            self.action_size = [env_info["action_space"].n]
            self.worker_size = self.env.worker_num
            self.env_type = "Multiworker"

        elif isinstance(self.env, gym.Env) or isinstance(self.env, gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.shape[0]]
            self.worker_size = 1
            self.env_type = "gym"

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
                    self.action_size,
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
                    self.action_size,
                    False,
                    self.prioritized_replay_eps,
                )

        else:
            if self.n_step_method:
                self.replay_buffer = NstepReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_size,
                    self.worker_size,
                    self.n_step,
                    self.gamma,
                )
            else:
                self.replay_buffer = ReplayBuffer(
                    self.buffer_size, self.observation_space, self.action_size
                )

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def actions(self, obs, steps):
        pass

    def run_name_update_with_tags(self, run_name):
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
        run_name="DPG_network",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        run_name = self.run_name_update_with_tags(run_name)
        self.eval_freq = total_timesteps // 100

        pbar = trange(total_timesteps, miniters=log_interval)
        with TensorboardWriter(self.log_dir, run_name) as (
            self.mlflowrun,
            self.save_path,
        ):
            if self.env_type == "gym":
                self.learn_gym(pbar, callback, log_interval)
                self.eval_gym(total_timesteps)
            if self.env_type == "Multiworker":
                self.learn_Multiworker(pbar, callback, log_interval)
            
            add_hparams(self, self.mlflowrun)
            self.save_params(self.save_path)

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"
        return discription

    def learn_gym(self, pbar, callback=None, log_interval=1000):
        state, info = self.env.reset()
        state = [np.expand_dims(state, axis=0)]
        self.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            actions = self.actions(state, steps)
            next_state, reward, terminated, truncated, info = self.env.step(actions[0])
            next_state = [np.expand_dims(next_state, axis=0)]
            self.replay_buffer.add(state, actions[0], reward, next_state, terminated, truncated)
            state = next_state
            if terminated or truncated:
                state, info = self.env.reset()
                state = [np.expand_dims(state, axis=0)]

            if steps > self.learning_starts and steps % self.train_freq == 0:
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval_gym(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def eval_gym(self, steps):
        total_reward = np.zeros(self.eval_eps)
        total_ep_len = np.zeros(self.eval_eps)
        total_truncated = np.zeros(self.eval_eps)

        state, info = self.eval_env.reset()
        state = [np.expand_dims(state, axis=0)]
        terminated = False
        truncated = False
        eplen = 0
        
        for ep in range(self.eval_eps):
            while not terminated and not truncated:
                actions = self.actions(
                    state, steps
                )
                next_state, reward, terminated, truncated, info = self.eval_env.step(actions[0])
                next_state = [np.expand_dims(next_state, axis=0)]
                total_reward[ep] += reward
                state = next_state
                eplen += 1

            total_ep_len[ep] = eplen
            total_truncated[ep] = float(truncated)

            state, info = self.eval_env.reset()
            state = [np.expand_dims(state, axis=0)]
            terminated = False
            truncated = False
            eplen = 0

        mean_reward = np.mean(total_reward)
        mean_ep_len = np.mean(total_ep_len)

        if self.mlflowrun:
            self.mlflowrun.log_metric("env/episode_reward", mean_reward, steps)
            self.mlflowrun.log_metric("env/episode len", mean_ep_len, steps)
            self.mlflowrun.log_metric("env/time over", np.mean(total_truncated), steps)
        return {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}

    def learn_Multiworker(self, pbar, callback=None, log_interval=1000):
        state, _, _, _, _, _ = self.env.get_steps()
        self.scores = np.zeros([self.worker_size])
        self.eplen = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            self.eplen += 1
            actions = self.actions([state], steps)
            self.env.step(actions)

            if steps > self.learning_starts and steps % self.train_freq == 0:
                loss = self.train_step(steps, self.gradient_steps)
                self.lossque.append(loss)

            (
                next_states,
                rewards,
                terminateds,
                truncateds,
                end_states,
                end_idx,
            ) = self.env.get_steps()
            nxtstates = np.copy(next_states)
            if end_states is not None:
                nxtstates[end_idx] = end_states
                if self.mlflowrun:
                    self.mlflowrun.log_metric(
                        "env/episode_reward", np.mean(self.scores[end_idx]), steps
                    )
                    self.mlflowrun.log_metric("env/episode len", np.mean(self.eplen[end_idx]), steps)
                    self.mlflowrun.log_metric(
                        "env/time over",
                        np.mean(truncateds[end_idx].astype(np.float32)),
                        steps,
                    )
                self.scoreque.extend(self.scores[end_idx])
                self.scores[end_idx] = 0
                self.eplen[end_idx] = 0
            self.replay_buffer.add([state], actions, rewards, [nxtstates], terminateds, truncateds)
            self.scores += rewards
            state = next_states

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description(self.discription())
        return np.mean(self.scoreque)

    def test(self, episode=10, run_name=None):
        if run_name is None:
            run_name = self.save_path

        directory = run_name
        if self.env_type == "gym":
            self.test_gym(episode, directory)

    def test_unity(self, episode, directory):
        pass

    def test_action(self, state):
        return self.actions(state, np.inf)

    def test_gym(self, episode, directory):
        from gymnasium.wrappers import RecordVideo

        Render_env = RecordVideo(self.env, directory, episode_trigger=lambda x: True)
        for i in range(episode):
            state, info = Render_env.reset()
            state = [np.expand_dims(state, axis=0)]
            terminated = False
            truncated = False
            episode_rew = 0
            while not (terminated or truncated):
                actions = self.test_action(state)
                observation, reward, terminated, truncated, info = Render_env.step(actions[0])
                state = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
            print("episod reward :", episode_rew)