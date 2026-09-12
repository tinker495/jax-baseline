from contextlib import nullcontext

import jax
import numpy as np

from jax_baselines.core.env_info import prepare_worker_env
from jax_baselines.core.env_protocols import (
    EvaluationContextEnv,
    VectorizedEvalEnv,
    batch_observation,
    log_environment_metrics,
    reset_for_evaluation,
    vector_autoreset_mask,
)


def log_measurement(
    log_metric,
    namespace,
    steps,
    *,
    episode_reward,
    episode_length,
    timeout_rate,
):
    """Write algorithm-level episode statistics under a namespace prefix."""
    log_metric(f"{namespace}/episode_reward", episode_reward, steps)
    log_metric(f"{namespace}/episode_length", episode_length, steps)
    log_metric(f"{namespace}/timeout_rate", timeout_rate, steps)


def _normalize_action_for_step(step_action):
    """Convert model output to an env.step-compatible action.

    Discrete action spaces produce an integer index, and several envs (e.g. ALE)
    require a native Python int, so a single integer action is returned as a
    Python scalar. Continuous (Box) action spaces produce float actions and the
    env expects an array even for a one-element Box: Pendulum indexes its action
    as ``np.clip(u, ...)[0]``, which fails on a Python float. Float actions are
    therefore always returned as a 1-D array. A leading batch dim of size 1
    (single-env rollout) is squeezed in both cases.
    """
    arr = step_action if isinstance(step_action, jax.Array) else np.asarray(step_action)
    if np.issubdtype(arr.dtype, np.integer) and arr.size == 1:
        return arr.item()
    return arr.reshape(-1)


def _evaluate_single_episodes(eval_env, eval_eps, act_eval_fn, conv_action, logger_run, steps):
    total_reward = np.zeros(eval_eps)
    total_ep_len = np.zeros(eval_eps)
    total_truncated = np.zeros(eval_eps)

    obs, _ = reset_for_evaluation(eval_env)
    obs = batch_observation(obs)
    terminated = False
    truncated = False
    eplen = 0

    for ep in range(eval_eps):
        while not terminated and not truncated:
            actions = act_eval_fn(obs)
            step_action = conv_action(actions) if conv_action is not None else actions

            # Normalize action so env.step receives a proper scalar when applicable
            action_to_step = _normalize_action_for_step(step_action)

            observation, reward, terminated, truncated, _ = eval_env.step(action_to_step)
            log_environment_metrics(eval_env, logger_run, steps, namespace="eval", flush=False)
            obs = batch_observation(observation)
            total_reward[ep] += reward
            eplen += 1

        total_ep_len[ep] = eplen
        total_truncated[ep] = float(truncated)
        obs, _ = eval_env.reset()
        obs = batch_observation(obs)
        terminated = False
        truncated = False
        eplen = 0

    return total_reward, total_ep_len, total_truncated


def _evaluate_vector_episodes(
    eval_env: VectorizedEvalEnv, eval_eps, act_eval_fn, conv_action, logger_run, steps
):
    workers = eval_env.get_info()["worker_num"]
    if workers < 1:
        raise ValueError("Evaluation worker count must be positive")
    # Fixed quotas prevent fast, short episodes from dominating the measurement.
    # ponytail: life quotas may omit Atari game scores; use game quotas for full-game eval.
    targets = (eval_eps + np.arange(workers)) // workers
    counts = np.zeros(workers, dtype=np.int64)
    rewards_sum = np.zeros(workers)
    lengths = np.zeros(workers, dtype=np.int64)
    eval_env.reset()
    prev_done = np.zeros(workers, dtype=bool)
    total_reward = np.zeros(eval_eps)
    total_ep_len = np.zeros(eval_eps)
    total_truncated = np.zeros(eval_eps)
    completed = 0

    while completed < eval_eps:
        # Keep the full batch, including workers that have exhausted their quotas.
        actions = act_eval_fn(eval_env.current_obs())
        eval_env.step(conv_action(actions) if conv_action is not None else actions)
        _, rewards, terminateds, truncateds, infos = eval_env.get_result()
        rewards, terminateds, truncateds, autoreset = jax.device_get(
            (
                rewards,
                terminateds,
                truncateds,
                vector_autoreset_mask(eval_env, terminateds, truncateds, infos),
            )
        )
        done = np.logical_or(terminateds, truncateds)
        active = ~prev_done & (counts < targets)
        log_environment_metrics(
            eval_env, logger_run, steps, namespace="eval", active=active, flush=False
        )
        rewards_sum[active] += rewards[active]
        lengths[active] += 1
        finished = done & active
        end = completed + int(finished.sum())
        total_reward[completed:end] = rewards_sum[finished]
        total_ep_len[completed:end] = lengths[finished]
        total_truncated[completed:end] = truncateds[finished]
        completed = end
        counts[finished] += 1
        rewards_sum[finished] = 0
        lengths[finished] = 0
        prev_done = done & autoreset & active

    return total_reward, total_ep_len, total_truncated


def evaluate_policy(eval_env, eval_eps, act_eval_fn, logger_run=None, steps=0, conv_action=None):
    """Measure exactly eval_eps episodes using the environment's native batch shape."""
    if eval_eps < 1:
        raise ValueError("eval_eps must be positive")
    collect = (
        _evaluate_vector_episodes
        if isinstance(eval_env, VectorizedEvalEnv)
        else _evaluate_single_episodes
    )
    with (
        eval_env.evaluation_context()
        if isinstance(eval_env, EvaluationContextEnv)
        else nullcontext()
    ):
        total_reward, total_ep_len, total_truncated = collect(
            eval_env, eval_eps, act_eval_fn, conv_action, logger_run, steps
        )
        log_environment_metrics(eval_env, logger_run, steps, namespace="eval")
    mean_reward = np.mean(total_reward)
    mean_ep_len = np.mean(total_ep_len)

    if logger_run is not None:
        log_measurement(
            logger_run.log_metric,
            "eval",
            steps,
            episode_reward=mean_reward,
            episode_length=mean_ep_len,
            timeout_rate=np.mean(total_truncated),
        )

    return {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}


def run_test_episodes(
    test_env, actions_eval_fn, episode, conv_action=None, *, logger_run=None, logging_env=None
):
    """Run evaluation episodes on an already-constructed test environment."""
    if episode < 1:
        raise ValueError("episode must be positive")
    # Recording wrappers step the same environment but need not expose its diagnostics.
    logging_env = test_env if logging_env is None else logging_env
    total_rewards = []
    for _ in range(episode):
        obs, _ = test_env.reset()
        obs = batch_observation(obs)
        terminated = False
        truncated = False
        episode_rew = 0
        eplen = 0
        while not terminated and not truncated:
            actions = actions_eval_fn(obs)
            step_action = conv_action(actions) if conv_action is not None else actions
            action_to_step = _normalize_action_for_step(step_action)

            observation, reward, terminated, truncated, _ = test_env.step(action_to_step)
            log_environment_metrics(logging_env, logger_run, None, namespace="test", flush=False)
            obs = batch_observation(observation)
            episode_rew += reward
            eplen += 1
        print("episod reward :", episode_rew, "episod len :", eplen)
        total_rewards.append(episode_rew)

    log_environment_metrics(logging_env, logger_run, None, namespace="test")
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"reward : {avg_reward} +- {std_reward}(std)")
    return avg_reward, std_reward


def record_and_test(env_builder, logger_run, actions_eval_fn, episode, conv_action=None):
    """Run an unrecorded evaluation loop when no experiments recorder is injected.

    Concrete Gymnasium ``RecordVideo`` / ``RecordEpisodeStatistics`` wrapping
    lives in ``experiments.runtime_adapters.record_and_test``. This fallback
    preserves the direct core ``agent.test()`` reward/std return shape without
    creating video artifacts or importing Gymnasium wrappers.
    """

    test_env, _ = prepare_worker_env(env_builder)
    try:
        return run_test_episodes(
            test_env, actions_eval_fn, episode, conv_action, logger_run=logger_run
        )
    finally:
        test_env.close()
