import base64
import multiprocessing as mp
import traceback
from functools import partial
from importlib import import_module

import jax
import numpy as np

from jax_baselines.core.env_info import prepare_worker_env
from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer
from jax_baselines.core.seeding import seed_prngs


class Impala_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    @classmethod
    def remote(cls, *args, **kwargs):
        return import_module("ray").remote(num_cpus=1)(cls).remote(*args, **kwargs)

    def __init__(self, env_builder, seed=None) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        seed_prngs(seed)
        # env_builder is the repo-local Environment Adapter callable injected by
        # experiments; the adapter prepares the env and normalized metadata.
        self.env, self.env_info = prepare_worker_env(env_builder, seed=seed)

    def get_info(self):
        return self.env_info

    def run(
        self,
        local_size,
        buffer_info,
        worker_replay_factory,
        model_builder,
        actor_builder,
        param_server,
        update,
        logger_server,
        stop,
        seed=None,
    ):
        try:
            ray = import_module("ray")
            seed_prngs(seed)
            queue, env_dict, actor_num = buffer_info
            local_buffer = make_worker_local_replay_buffer(
                worker_replay_factory, local_size, env_dict, None
            )
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
            rw_label = "rollout/episode_reward"
            if have_original_reward:
                original_rw_label = "rollout/original_reward"
            len_label = "rollout/episode_length"
            to_label = "rollout/timeout_rate"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                for _ in range(local_size):
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
        except Exception:
            print(
                "------------------------------Exception in worker----------------------------------"
            )
            traceback.print_exc()
            print(
                "---------------------------------------------------------------------------------"
            )
        finally:
            if stop.is_set():
                print("worker stoped")
            else:
                stop.set()
        return None
