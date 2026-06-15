import base64
import multiprocessing as mp
from functools import partial
from importlib import import_module

import jax
import numpy as np

from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer
from jax_baselines.core.seeding import seed_prngs


class Ape_X_Worker(object):
    encoded = base64.b64encode(mp.current_process().authkey)

    @classmethod
    def remote(cls, *args, **kwargs):
        return import_module("ray").remote(num_cpus=1)(cls).remote(*args, **kwargs)

    def __init__(self, env_name_, seed=None) -> None:
        mp.current_process().authkey = base64.b64decode(self.encoded)
        atari_wrappers = import_module("env_builder.atari_wrappers")
        get_env_type = atari_wrappers.get_env_type
        make_wrap_atari = atari_wrappers.make_wrap_atari
        seed_env = import_module("env_builder.seeding").seed_env

        seed_prngs(seed)

        # Accept either an env id (string) or an env_builder callable
        if callable(env_name_):
            # env_builder is expected to return a gym env when called
            try:
                env = env_name_(worker=1, seed=seed)
            except TypeError:
                try:
                    env = env_name_(1, seed=seed)
                except TypeError:
                    # some builders accept worker/render args; try default worker=1
                    env = env_name_()
            self.env = env
            # Try to infer env_id from env.spec
            try:
                self.env_id = env.unwrapped.spec.id
            except Exception:
                self.env_id = getattr(getattr(env, "spec", None), "id", None)
            # If we can, map env_id to env_type, otherwise fallback to SingleEnv
            if self.env_id is not None:
                self.env_type, _ = get_env_type(self.env_id)
                if self.env_type is None:
                    self.env_type = "SingleEnv"
            else:
                # Best-effort fallback
                self.env_type = "SingleEnv"
        else:
            self.env_type, self.env_id = get_env_type(env_name_)
            if self.env_type == "atari_env":
                self.env = make_wrap_atari(env_name_, clip_rewards=True)
            else:
                gym = import_module("gymnasium")
                self.env = gym.make(env_name_)
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
        worker_replay_factory,
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
            ray = import_module("ray")
            seed_prngs(seed)
            global_buffer, env_dict, n_s = buffer_info
            local_buffer = make_worker_local_replay_buffer(
                worker_replay_factory, local_size, env_dict, n_s
            )
            preproc, model = model_builder()
            (
                get_abs_td_error,
                actor,
                get_action,
                random_action,
                key_seq,
            ) = actor_builder()

            get_abs_td_error = jax.jit(partial(get_abs_td_error, model, preproc))
            actor = jax.jit(partial(actor, model, preproc))
            _get_action = partial(get_action, actor)
            get_action = random_action

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
            params = ray.get(param_server.get_params.remote())
            eplen = 0
            episode = 0
            if eps is None:
                rw_label = "rollout/episode_reward"
                if have_original_reward:
                    original_rw_label = "rollout/original_reward"
                len_label = "rollout/episode_length"
                to_label = "rollout/timeout_rate"
            else:
                rw_label = f"rollout/episode_reward/eps{eps:.2f}"
                if have_original_reward:
                    original_rw_label = f"rollout/original_reward/eps{eps:.2f}"
                len_label = f"rollout/episode_length/eps{eps:.2f}"
                to_label = f"rollout/timeout_rate/eps{eps:.2f}"

            while not stop.is_set():
                if update.is_set():
                    params = ray.get(param_server.get_params.remote())
                    update.clear()
                    get_action = _get_action

                eplen += 1
                actions = get_action(params, obs, eps, next(key_seq))
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                next_obs = [np.expand_dims(next_obs, axis=0)]
                local_buffer.add(obs, actions, reward, next_obs, terminated or truncated, truncated)
                if have_original_reward:
                    original_score += info["original_reward"]
                score += reward
                obs = next_obs

                if terminated or truncated:
                    local_buffer.episode_end()
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

                if len(local_buffer) >= local_size:
                    transition = local_buffer.get_buffer()
                    local_buffer.clear()
                    abs_td_error = get_abs_td_error(
                        params,
                        **local_buffer.conv_transitions(transition),
                        key=next(key_seq),
                    )
                    global_buffer.add(**transition, priorities=abs_td_error)
        finally:
            if stop.is_set():
                print("worker stopped")
            else:
                stop.set()
        return None
