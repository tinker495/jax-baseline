from functools import partial

import jax

from jax_baselines.core.env_info import prepare_worker_env
from jax_baselines.core.env_protocols import batch_observation
from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer
from jax_baselines.core.seeding import seed_prngs


class Ape_X_Worker(object):
    def __init__(self, env_builder, seed=None) -> None:
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
        logger_server,
        update,
        stop,
        eps=0.05,
        seed=None,
    ):
        try:
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
            obs = batch_observation(obs)
            params = param_server.get_params()
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
                    params = param_server.get_params()
                    update.clear()
                    get_action = _get_action

                eplen += 1
                actions = get_action(params, obs, eps, next(key_seq))
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                next_obs = batch_observation(next_obs)
                local_buffer.add(obs, actions, reward, next_obs, terminated, truncated)
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
                            to_label: float(truncated),
                        }
                        if have_original_reward:
                            if have_lives:
                                if info["lives"] == 0:
                                    log_dict[original_rw_label] = original_score
                                    original_score = 0
                            else:
                                log_dict[original_rw_label] = original_score
                                original_score = 0
                        logger_server.log_worker(log_dict, episode)
                    score = 0
                    eplen = 0
                    episode += 1
                    obs, info = self.env.reset()
                    obs = batch_observation(obs)

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
