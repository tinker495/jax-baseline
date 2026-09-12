from functools import partial

import jax

from jax_baselines.APE_X.common_servers import WorkerMetricLogger
from jax_baselines.core.env_info import prepare_worker_env
from jax_baselines.core.env_protocols import batch_observation, log_environment_metrics
from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer
from jax_baselines.core.seeding import seed_prngs


class Impala_Worker:
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
        update,
        logger_server,
        stop,
        seed=None,
    ):
        try:
            seed_prngs(seed)
            queue, env_dict, _actor_num = buffer_info
            local_buffer = make_worker_local_replay_buffer(
                worker_replay_factory, local_size, env_dict, None
            )
            actor_model, _ = model_builder()
            actor, get_action_prob, convert_action = actor_builder()

            actor = jax.jit(partial(actor, actor_model))
            get_action_prob = partial(get_action_prob, actor)

            if seed is not None:
                try:
                    obs, _info = self.env.reset(seed=seed)
                except TypeError:
                    obs, _info = self.env.reset()
            else:
                obs, _info = self.env.reset()
            score = 0
            obs = batch_observation(obs)
            eplen = 0
            episode = 0
            environment_logger = WorkerMetricLogger()
            rw_label = "rollout/episode_reward"
            len_label = "rollout/episode_length"
            to_label = "rollout/timeout_rate"

            # Eager initial fetch so actor parameters are always bound before first use,
            # mirroring the APE-X workers (avoids reliance on update being pre-set).
            actor_params = jax.device_put(param_server.get_params())
            while not stop.is_set():
                if update.is_set():
                    actor_params = jax.device_put(param_server.get_params())
                    update.clear()
                for _ in range(local_size):
                    eplen += 1
                    actions, log_prob = get_action_prob(actor_params, obs)
                    next_obs, reward, terminated, truncated, _info = self.env.step(
                        convert_action(actions)
                    )
                    next_obs = batch_observation(next_obs)
                    local_buffer.add(
                        obs,
                        actions,
                        log_prob,
                        reward,
                        next_obs,
                        terminated,
                        truncated,
                    )
                    if logger_server is not None:
                        log_environment_metrics(
                            self.env,
                            environment_logger,
                            episode,
                            flush=bool(terminated or truncated),
                        )
                    score += reward
                    obs = next_obs

                    if terminated or truncated:
                        if logger_server is not None:
                            log_dict = {
                                **environment_logger.metrics,
                                rw_label: score,
                                len_label: eplen,
                                to_label: float(truncated),
                            }
                            logger_server.log_worker(log_dict, episode)
                            environment_logger.metrics.clear()
                        score = 0
                        eplen = 0
                        episode += 1
                        obs, _info = self.env.reset()
                        obs = batch_observation(obs)
                queue.put(local_buffer.get_buffer())
            if logger_server is not None:
                log_environment_metrics(self.env, environment_logger, episode)
                if environment_logger.metrics:
                    logger_server.log_worker(environment_logger.metrics, episode)
        finally:
            if stop.is_set():
                print("worker stopped")
            else:
                stop.set()
