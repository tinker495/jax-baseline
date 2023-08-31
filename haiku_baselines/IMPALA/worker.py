import base64
import multiprocessing as mp
import time
import jax
import jax.numpy as jnp
from functools import partial
import os, psutil
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray
import haiku as hk

from haiku_baselines.IMPALA.cpprb_buffers import EpochBuffer


@ray.remote(num_cpus=1)
class Impala_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    def __init__(self, env_name_) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        from haiku_baselines.common.atari_wrappers import make_wrap_atari, get_env_type

        self.env_type, self.env_id = get_env_type(env_name_)
        if self.env_type == "atari_env" and not "MinAtar" in env_name_:
            self.env = make_wrap_atari(env_name_, clip_rewards=True)
        else:
            self.env = gym.make(env_name_)

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
        network_builder,
        actor_builder,
        param_server,
        update,
        logger_server,
        stop,
    ):
        try:
            queue, env_dict, actor_num = buffer_info
            local_buffer = EpochBuffer(local_size, env_dict)
            preproc, model, _ = network_builder()
            actor, get_action_prob, convert_action = actor_builder()

            actor = jax.jit(partial(actor, model, preproc))
            get_action_prob = partial(get_action_prob, actor)

            score = 0
            state, info = self.env.reset()
            state = [np.expand_dims(state, axis=0)]
            eplen = 0
            episode = 0
            rw_label = f"env/episode_reward"
            len_label = f"env/episode_len"
            to_label = f"env/time_over"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                for i in range(local_size):
                    eplen += 1
                    actions, log_prob = get_action_prob(params, state)
                    next_state, reward, terminal, truncated, info = self.env.step(
                        convert_action(actions)
                    )
                    next_state = [np.expand_dims(next_state, axis=0)]
                    local_buffer.add(
                        state,
                        actions,
                        log_prob,
                        reward,
                        next_state,
                        terminal or truncated,
                        truncated,
                    )
                    score += reward
                    state = next_state

                    if terminal or truncated:
                        state, info = self.env.reset()
                        state = [np.expand_dims(state, axis=0)]
                        if logger_server is not None:
                            log_dict = {
                                rw_label: score,
                                len_label: eplen,
                                to_label: 1 - terminal,
                            }
                            logger_server.log_worker.remote(log_dict, episode)
                        score = 0
                        eplen = 0
                        episode += 1
                queue.put_nowait(local_buffer.get_buffer())
        finally:
            if stop.is_set():
                print("worker stoped")
            else:
                stop.set()
        return None
