import numpy as np

from jax_baselines.core.env_info import prepare_worker_env


def extract_original_reward(info):
    """Return a single-step original reward from an env info dict, if present."""
    if isinstance(info, dict):
        return info.get("original_reward")
    return None


def extract_lives(info):
    """Return a single-step lives count from an env info dict, if present."""
    if isinstance(info, dict):
        return info.get("lives")
    return None


def _extract_vector_info_values(infos, worker_size, key, dtype):
    values = np.zeros(worker_size, dtype=np.float64)
    present = np.zeros(worker_size, dtype=bool)

    if isinstance(infos, dict):
        if key not in infos:
            return values, present
        raw = np.asarray(infos[key], dtype=dtype)
        if raw.shape == ():
            raw = np.full(worker_size, raw.item(), dtype=dtype)
        else:
            raw = raw.reshape(-1)
        count = min(worker_size, raw.shape[0])
        values[:count] = raw[:count]
        mask = infos.get(f"_{key}")
        if mask is None:
            present[:count] = True
        else:
            present[:count] = np.asarray(mask, dtype=bool).reshape(-1)[:count]
        return values, present

    if isinstance(infos, (list, tuple)):
        for idx, info in enumerate(infos[:worker_size]):
            if isinstance(info, dict) and key in info:
                values[idx] = info[key]
                present[idx] = True
        return values, present

    return values, present


def extract_vector_original_rewards(infos, worker_size):
    """Return per-worker original rewards and a presence mask from vector infos.

    Gymnasium vector envs usually return a dict of arrays, while some wrappers
    expose one info dict per worker. Supporting both shapes keeps rollout
    logging independent from the vector backend.
    """
    return _extract_vector_info_values(infos, worker_size, "original_reward", np.float64)


def extract_vector_lives(infos, worker_size):
    """Return per-worker lives counts and a presence mask from vector infos."""
    return _extract_vector_info_values(infos, worker_size, "lives", np.int32)


def log_measurement(
    log_metric,
    namespace,
    steps,
    *,
    episode_reward,
    episode_length,
    timeout_rate,
    original_reward=None,
):
    """Write the canonical four measurement leaves under a namespace prefix.

    The single tag-writer for both the ``eval/`` (frozen policy measured on a
    separate ``eval_env``) and ``rollout/`` (behavior policy's own training
    episodes) namespaces, so their leaf names can never drift apart again.
    ``original_reward`` is logged only when an Atari wrapper supplied an
    unclipped score.
    """
    if original_reward is not None:
        log_metric(f"{namespace}/original_reward", original_reward, steps)
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
    arr = np.asarray(step_action)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = np.asarray(arr[0])
    if np.issubdtype(arr.dtype, np.integer) and arr.size == 1:
        return arr.reshape(-1)[0].item()
    return np.reshape(arr, (-1,))


def evaluate_policy(eval_env, eval_eps, act_eval_fn, logger_run=None, steps=0, conv_action=None):
    """General evaluation helper used by multiple base classes.

    act_eval_fn: callable(obs) -> action (already formatted for step)
    conv_action: optional function to convert action before stepping (A2C)
    """
    original_rewards = []
    total_reward = np.zeros(eval_eps)
    total_ep_len = np.zeros(eval_eps)
    total_truncated = np.zeros(eval_eps)

    # Prefer a true environment reset if available (e.g., Atari EpisodicLifeEnv)
    if hasattr(eval_env, "true_reset") and callable(eval_env.true_reset):
        obs, info = eval_env.true_reset()
    else:
        obs, info = eval_env.reset()
    obs = [np.expand_dims(obs, axis=0)]
    have_original_reward = "original_reward" in info
    have_lives = "lives" in info
    if have_original_reward:
        original_reward = info["original_reward"]
    terminated = False
    truncated = False
    eplen = 0

    for ep in range(eval_eps):
        while not terminated and not truncated:
            actions = act_eval_fn(obs)
            step_action = conv_action(actions) if conv_action is not None else actions

            # Normalize action so env.step receives a proper scalar when applicable
            action_to_step = _normalize_action_for_step(step_action)

            observation, reward, terminated, truncated, info = eval_env.step(action_to_step)
            obs = [np.expand_dims(observation, axis=0)]
            if have_original_reward:
                original_reward += info.get("original_reward", 0)
            total_reward[ep] += reward
            eplen += 1

        total_ep_len[ep] = eplen
        total_truncated[ep] = float(truncated)
        if have_original_reward:
            if have_lives:
                if info.get("lives", 0) == 0:
                    original_rewards.append(original_reward)
                    original_reward = 0
            else:
                original_rewards.append(original_reward)
                original_reward = 0

        obs, info = eval_env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        terminated = False
        truncated = False
        eplen = 0

    mean_reward = np.mean(total_reward)
    mean_ep_len = np.mean(total_ep_len)

    mean_original_score = None
    if have_original_reward and len(original_rewards) > 0:
        mean_original_score = np.mean(original_rewards)

    if logger_run:
        log_measurement(
            logger_run.log_metric,
            "eval",
            steps,
            episode_reward=mean_reward,
            episode_length=mean_ep_len,
            timeout_rate=np.mean(total_truncated),
            original_reward=mean_original_score,
        )

    if mean_original_score is not None:
        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
            "mean_original_score": mean_original_score,
        }
    else:
        return {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}


def run_test_episodes(test_env, actions_eval_fn, episode, conv_action=None):
    """Run evaluation episodes on an already-constructed test environment."""

    total_rewards = []
    for _ in range(episode):
        obs, info = test_env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        terminated = False
        truncated = False
        episode_rew = 0
        eplen = 0
        while not terminated and not truncated:
            actions = actions_eval_fn(obs)
            step_action = conv_action(actions) if conv_action is not None else actions
            action_to_step = _normalize_action_for_step(step_action)

            observation, reward, terminated, truncated, info = test_env.step(action_to_step)
            obs = [np.expand_dims(observation, axis=0)]
            episode_rew += reward
            eplen += 1
        print("episod reward :", episode_rew, "episod len :", eplen)
        total_rewards.append(episode_rew)

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
        return run_test_episodes(test_env, actions_eval_fn, episode, conv_action)
    finally:
        close = getattr(test_env, "close", None)
        if callable(close):
            close()
