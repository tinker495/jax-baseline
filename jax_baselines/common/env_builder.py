from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import ray
from gymnasium import spaces


def get_env_builder(env_name, **kwargs):
    def env_builder(worker=1, render_mode=None):
        if worker > 1:
            return rayVectorizedGymEnv(env_name, worker_num=worker)
        else:
            from jax_baselines.common.atari_wrappers import (
                get_env_type,
                make_wrap_atari,
            )

            env_type, env_id = get_env_type(env_name)
            if env_type == "atari_env":
                env = make_wrap_atari(env_name, clip_rewards=True)
            else:
                env = gym.make(env_name, render_mode=render_mode)
            return env

    env_type = "SingleEnv"
    env_info = {
        "env_type": env_type,
        "env_id": env_name,
    }
    return env_builder, env_info


class ActionSpace:
    pass


class ObservationSpace:
    pass


class Env(ABC):
    class EnvInfo:
        env_type: str
        env_id: str
        num_workers: int
        observation_space: list[tuple[int, ...]]
        action_space: list[tuple[int, ...]]

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def current_obs(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def close(self):
        pass


class SingleEnv(Env):

    env_info = None
    num_workers = 1


class VectorizedEnv(Env):

    env_info = None
    num_workers = None


class rayVectorizedGymEnv(VectorizedEnv):
    def __init__(self, env_id, worker_num=8, render=False):
        ray.init(num_cpus=worker_num)
        self.env_id = env_id
        self.worker_num = worker_num
        self.workers = [
            gymRayworker.remote(env_id, render=(w == 0) if render else False)
            for w in range(worker_num)
        ]
        self.env_info = ray.get(self.workers[0].get_info.remote())
        resets = ray.get([w.get_reset.remote() for w in self.workers])
        obs_list, self.reset_info = zip(*resets)
        self.obs = np.stack(obs_list, axis=0)

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def step(self, actions):
        self.steps = [w.step.remote(a) for w, a in zip(self.workers, actions)]

    def get_result(self):
        steps = ray.get(self.steps)
        next_obs, end_states, rewards, terminateds, truncateds, dones, infos = zip(*steps)
        next_obs = np.stack(next_obs, axis=0)
        rewards = np.stack(rewards, axis=0)
        terminateds = np.stack(terminateds, axis=0)
        truncateds = np.stack(truncateds, axis=0)
        if any(dones):
            self.obs = np.copy(next_obs)
            for idx, done in enumerate(dones):
                if done:
                    next_obs[idx] = end_states[idx]
        else:
            self.obs = next_obs
        return next_obs, rewards, terminateds, truncateds, infos

    def close(self):
        ray.shutdown()


@ray.remote
class gymRayworker:
    def __init__(self, env_name_, render=False):
        from jax_baselines.common.atari_wrappers import get_env_type, make_wrap_atari

        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == "atari_env":
            self.env = make_wrap_atari(env_name_, clip_rewards=True)
        else:
            self.env = gym.make(env_name_)
        if not isinstance(self.env.action_space, spaces.Box):
            self.action_conv = lambda a: a[0]
        else:
            self.action_conv = lambda a: a
        self.render = render

    def get_reset(self):
        obs, info = self.env.reset()
        return obs, info

    def get_info(self):
        return {
            "observation_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "env_type": self.env_type,
            "env_id": self.env_id,
        }

    def step(self, action):
        if self.render:
            self.env.render()
        obs, reward, terminated, truncated, info = self.env.step(self.action_conv(action))
        done = terminated or truncated
        if done:
            done_obs = obs
            obs, _ = self.env.reset()
        else:
            done_obs = None
        return obs, done_obs, reward, terminated, truncated, done, info
