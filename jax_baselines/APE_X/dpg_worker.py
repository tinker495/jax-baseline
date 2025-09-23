import base64
import multiprocessing as mp
import traceback
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import ray
from jax_baselines.common.utils import seed_env, seed_prngs

from jax_baselines.common.cpprb_buffers import ReplayBuffer


@ray.remote(num_cpus=1)
class Ape_X_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    def __init__(self, env_name_, seed=None) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        self.env_type = "SingleEnv"
        self.env_id = None
        seed_prngs(seed)

        if callable(env_name_):
            try:
                env = env_name_(worker=1, seed=seed)
            except TypeError:
                try:
                    env = env_name_(1, seed=seed)
                except TypeError:
                    env = env_name_()
            self.env = env
            try:
                self.env_id = env.unwrapped.spec.id
            except Exception:
                self.env_id = getattr(getattr(env, "spec", None), "id", None)
        else:
            self.env = gym.make(env_name_)
            self.env_id = env_name_
        seed_env(self.env, seed)
        self.base_seed = seed

    def get_info(self):
        return {
            "observation_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "env_type": self.env_type,
            "env_id": self.env_id,
        }

    def run(
        self,
        local_size,
        buffer_info,
        model_builder,
        actor_builder,
        param_server,
        logger_server,
        update,
        stop,
        eps=0.05,
        seed=None,
    ):
        try:
            seed_prngs(seed)
            gloabal_buffer, env_dict, n_s = buffer_info
            local_buffer = ReplayBuffer(local_size, env_dict=env_dict, n_s=n_s)
            preproc, actor_model, cricit_model = model_builder()
            (
                get_abs_td_error,
                actor,
                get_action,
                random_action,
                noise,
                key_seq,
            ) = actor_builder()

            get_abs_td_error = jax.jit(
                partial(get_abs_td_error, actor_model, cricit_model, preproc)
            )
            actor = jax.jit(partial(actor, actor_model, preproc))
            _get_action = partial(get_action, actor)
            get_action = random_action

            score = 0
            if seed is not None:
                try:
                    obs, info = self.env.reset(seed=seed)
                except TypeError:
                    obs, info = self.env.reset()
            else:
                obs, info = self.env.reset()
            obs = [np.expand_dims(obs, axis=0)]
            params = ray.get(param_server.get_params.remote())
            eplen = 0
            episode = 0
            if eps is None:
                rw_label = "env/episode_reward"
                len_label = "env/episode_len"
                to_label = "env/time_over"
            else:
                rw_label = f"env/episode_reward/eps{eps:.2f}"
                len_label = f"env/episode_len/eps{eps:.2f}"
                to_label = f"env/time_over/eps{eps:.2f}"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                    get_action = _get_action

                eplen += 1
                actions = get_action(params, obs, noise, eps, next(key_seq))
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                next_obs = [np.expand_dims(next_obs, axis=0)]
                local_buffer.add(obs, actions, reward, next_obs, terminated or truncated, truncated)
                score += reward
                obs = next_obs

                if terminated or truncated:
                    local_buffer.episode_end()
                    obs, info = self.env.reset()
                    obs = [np.expand_dims(obs, axis=0)]
                    if logger_server is not None:
                        log_dict = {
                            rw_label: score,
                            len_label: eplen,
                            to_label: 1 - terminated,
                        }
                        logger_server.log_worker.remote(log_dict, episode)
                    score = 0
                    eplen = 0
                    episode += 1
                    noise.reset([0])

                if len(local_buffer) >= local_size:
                    transition = local_buffer.get_buffer()
                    local_buffer.clear()
                    abs_td_error = get_abs_td_error(
                        params,
                        **local_buffer.conv_transitions(transition),
                        key=next(key_seq),
                    )
                    gloabal_buffer.add(**transition, priorities=abs_td_error)
        except Exception as e:
            print(
                "------------------------------Exception in worker----------------------------------"
            )
            traceback.print_exc(e)
            print(
                "---------------------------------------------------------------------------------"
            )
        finally:
            if stop.is_set():
                print("worker stoped")
            else:
                stop.set()
        return None
