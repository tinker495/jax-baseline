import base64
import multiprocessing as mp
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import ray

from jax_baselines.common.utils import seed_env, seed_prngs

from jax_baselines.IMPALA.cpprb_buffers import EpochBuffer


@ray.remote(num_cpus=1)
class Impala_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    def __init__(self, env_name_, seed=None) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        from jax_baselines.common.atari_wrappers import get_env_type, make_wrap_atari

        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == "atari_env" and "MinAtar" not in env_name_:
            self.env = make_wrap_atari(env_name_, clip_rewards=True)
        else:
            self.env = gym.make(env_name_)
        self.base_seed = seed
        seed_prngs(seed)
        seed_env(self.env, seed)

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
        update,
        logger_server,
        stop,
        seed=None,
    ):
        try:
            seed_prngs(seed)
            queue, env_dict, actor_num = buffer_info
            local_buffer = EpochBuffer(local_size, env_dict)
            preproc, actor_model, _ = model_builder()
            actor, get_action_prob, convert_action = actor_builder()

            actor = jax.jit(partial(actor, actor_model, preproc))
            get_action_prob = partial(get_action_prob, actor)

            if seed is not None:
                try:
                    obs, info = self.env.reset(seed=seed)
                except TypeError:
                    obs, info = self.env.reset()
            else:
                obs, info = self.env.reset()
            have_original_reward = "original_reward" in info.keys()
            have_lives = "lives" in info.keys()
            if have_original_reward:
                original_score = 0
            score = 0
            obs = [np.expand_dims(obs, axis=0)]
            eplen = 0
            episode = 0
            rw_label = "env/episode_reward"
            if have_original_reward:
                original_rw_label = "env/original_reward"
            len_label = "env/episode_len"
            to_label = "env/time_over"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                for i in range(local_size):
                    eplen += 1
                    actions, log_prob = get_action_prob(params, obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        convert_action(actions)
                    )
                    next_obs = [np.expand_dims(next_obs, axis=0)]
                    local_buffer.add(
                        obs,
                        actions,
                        log_prob,
                        reward,
                        next_obs,
                        terminated or truncated,
                        truncated,
                    )
                    if have_original_reward:
                        original_score += info["original_reward"]
                    score += reward
                    obs = next_obs

                    if terminated or truncated:
                        if logger_server is not None:
                            log_dict = {
                                rw_label: score,
                                len_label: eplen,
                                to_label: 1 - terminated,
                            }
                            if have_original_reward:
                                if have_lives:
                                    if info["lives"] == 0:
                                        log_dict[original_rw_label] = original_score
                                        original_score = 0
                                else:
                                    log_dict[original_rw_label] = original_score
                                    original_score = 0
                            logger_server.log_worker.remote(log_dict, episode)
                        score = 0
                        eplen = 0
                        episode += 1
                        obs, info = self.env.reset()
                        obs = [np.expand_dims(obs, axis=0)]
                queue.put(local_buffer.get_buffer())
        except Exception as e:
            print(f"worker {mp.current_process().name} error : {e}")
        finally:
            if stop.is_set():
                print("worker stoped")
            else:
                stop.set()
        return None
