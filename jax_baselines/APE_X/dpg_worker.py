from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.APE_X.common_servers import WorkerMetricLogger
from jax_baselines.core.env_info import prepare_worker_env
from jax_baselines.core.env_protocols import batch_observation, log_environment_metrics
from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer
from jax_baselines.core.seeding import seed_prngs
from jax_baselines.math.jax_utils import convert_normalized_obs


def make_behavior_action(actor_model, noise_step, epsilon):
    """One worker's policy action as one compiled call; PRNG key and noise are device carries.

    The policy action gets ``epsilon``-scaled exploration noise from
    ``noise_step(noise, key, reset)`` and is clipped to [-1, 1]. ``reset`` is a static,
    host-scheduled flag marking the first policy action of an episode.
    """

    def act(params, obs, key, noise, reset):
        key, noise_key = jax.random.split(key)
        noise = noise_step(noise, noise_key, reset)
        actions = actor_model(params["policy"], None, convert_normalized_obs(obs))
        return jnp.clip(actions + noise * epsilon, -1.0, 1.0)[0], key, noise

    return jax.jit(act, static_argnames="reset")


class Ape_X_Worker:
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
            actor_model, critic_model = model_builder()
            get_abs_td_error, noise_step = actor_builder()
            get_abs_td_error = partial(get_abs_td_error, actor_model, critic_model)

            @jax.jit
            def td_error(params, batch, key):
                key, subkey = jax.random.split(key)
                return get_abs_td_error(params, **batch, key=subkey), key

            act = make_behavior_action(actor_model, noise_step, eps)
            # Drawn from the NumPy stream seed_prngs just seeded (unseeded runs stay random).
            key = jax.random.PRNGKey(jax.device_put(np.random.randint(np.iinfo(np.int32).max)))
            noise = jax.device_put(np.zeros((1, self.env_info["action_size"][0]), np.float32))
            policy_ready = False
            # Pending until a policy action consumes it: warmup actions carry no noise.
            reset_noise = True

            score = 0
            if seed is not None:
                try:
                    obs, _info = self.env.reset(seed=seed)
                except TypeError:
                    obs, _info = self.env.reset()
            else:
                obs, _info = self.env.reset()
            obs = batch_observation(obs)
            params = jax.device_put(param_server.get_params())
            eplen = 0
            episode = 0
            pending_steps = 0
            environment_logger = WorkerMetricLogger("" if eps is None else f"/eps{eps:.2f}")
            if eps is None:
                rw_label = "rollout/episode_reward"
                len_label = "rollout/episode_length"
                to_label = "rollout/timeout_rate"
            else:
                rw_label = f"rollout/episode_reward/eps{eps:.2f}"
                len_label = f"rollout/episode_length/eps{eps:.2f}"
                to_label = f"rollout/timeout_rate/eps{eps:.2f}"

            while not stop.is_set():
                if update.is_set():
                    params = jax.device_put(param_server.get_params())
                    update.clear()
                    policy_ready = True

                eplen += 1
                if policy_ready:
                    # Env boundary: one explicit upload of the observation, one download.
                    actions, key, noise = act(
                        params, jax.device_put(obs), key, noise, reset=reset_noise
                    )
                    actions = jax.device_get(actions)
                    reset_noise = False
                else:
                    # Warmup actions need no device data: a host draw avoids the round trip.
                    actions = np.random.uniform(-1.0, 1.0, size=noise.shape[1:])
                next_obs, reward, terminated, truncated, _info = self.env.step(actions)
                pending_steps += 1
                next_obs = batch_observation(next_obs)
                local_buffer.add(obs, actions, reward, next_obs, terminated, truncated)
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
                    local_buffer.episode_end()
                    obs, _info = self.env.reset()
                    obs = batch_observation(obs)
                    if logger_server is not None:
                        log_dict = {
                            **environment_logger.metrics,
                            rw_label: score,
                            len_label: eplen,
                            to_label: float(truncated),
                        }
                        logger_server.log_worker(log_dict, episode, environment_steps=pending_steps)
                        pending_steps = 0
                        environment_logger.metrics.clear()
                    score = 0
                    eplen = 0
                    episode += 1
                    reset_noise = True

                if len(local_buffer) >= local_size:
                    transition = local_buffer.get_buffer()
                    local_buffer.clear()
                    abs_td_error, key = td_error(
                        params, jax.device_put(local_buffer.conv_transitions(transition)), key
                    )
                    global_buffer.add(**transition, priorities=jax.device_get(abs_td_error))
                    if logger_server is not None and pending_steps:
                        logger_server.log_worker({}, episode, environment_steps=pending_steps)
                        pending_steps = 0
            if logger_server is not None:
                log_environment_metrics(self.env, environment_logger, episode)
                logger_server.log_worker(
                    environment_logger.metrics,
                    episode,
                    environment_steps=pending_steps,
                    flush=True,
                )
        finally:
            if stop.is_set():
                print("worker stopped")
            else:
                stop.set()
