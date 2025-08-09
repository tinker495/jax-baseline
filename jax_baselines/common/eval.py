import numpy as np


def evaluate_policy(eval_env, eval_eps, act_eval_fn, logger_run=None, steps=0, conv_action=None):
    """General evaluation helper used by multiple base classes.

    act_eval_fn: callable(obs) -> action (already formatted for step)
    conv_action: optional function to convert action before stepping (A2C)
    """
    original_rewards = []
    total_reward = np.zeros(eval_eps)
    total_ep_len = np.zeros(eval_eps)
    total_truncated = np.zeros(eval_eps)

    obs, info = eval_env.reset()
    obs = [np.expand_dims(obs, axis=0)]
    have_original_reward = "original_reward" in info.keys()
    have_lives = "lives" in info.keys()
    if have_original_reward:
        original_reward = info["original_reward"]
    terminated = False
    truncated = False
    eplen = 0

    for ep in range(eval_eps):
        while not terminated and not truncated:
            actions = act_eval_fn(obs)
            if conv_action is not None:
                step_action = conv_action(actions)
            else:
                step_action = actions

            # Normalize action to a Python scalar int when possible so downstream
            # env.step receives a native Python int (required by some envs like ALE).
            arr = np.asarray(step_action)
            # Scalars
            if arr.ndim == 0:
                action_to_step = arr.item()
            # 1-D arrays: could be a single discrete action [a] or a continuous
            # action vector [a1, a2, ...]. If size == 1 treat as scalar.
            elif arr.ndim == 1:
                if arr.size == 1:
                    action_to_step = arr.item()
                else:
                    action_to_step = arr
            else:
                # Higher-dim arrays: assume first entry corresponds to the
                # environment's action (batch dim). Extract the first element
                # and further reduce if necessary.
                first = arr[0]
                if np.asarray(first).ndim == 0:
                    action_to_step = first.item()
                else:
                    action_to_step = np.asarray(first)

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
        if mean_original_score is not None:
            logger_run.log_metric("env/original_reward", mean_original_score, steps)
        logger_run.log_metric("env/episode_reward", mean_reward, steps)
        logger_run.log_metric("env/episode len", mean_ep_len, steps)
        logger_run.log_metric("env/time over", np.mean(total_truncated), steps)

    if mean_original_score is not None:
        return {
            "mean_reward": mean_reward,
            "mean_ep_len": mean_ep_len,
            "mean_original_score": mean_original_score,
        }
    else:
        return {"mean_reward": mean_reward, "mean_ep_len": mean_ep_len}


def record_and_test(env_builder, logger_run, actions_eval_fn, episode, conv_action=None):
    import os

    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

    directory = logger_run.get_local_path("video")
    os.makedirs(directory, exist_ok=True)
    test_env = env_builder(1, render_mode="rgb_array")
    Render_env = RecordVideo(test_env, directory, episode_trigger=lambda x: True)
    Render_env = RecordEpisodeStatistics(Render_env)
    total_rewards = []
    with Render_env:
        for i in range(episode):
            obs, info = Render_env.reset()
            obs = [np.expand_dims(obs, axis=0)]
            terminated = False
            truncated = False
            episode_rew = 0
            eplen = 0
            while not terminated and not truncated:
                actions = actions_eval_fn(obs)
                if conv_action is not None:
                    step_action = conv_action(actions)
                else:
                    step_action = actions

                # Normalize action similar to evaluate_policy
                arr = np.asarray(step_action)
                if arr.ndim == 0:
                    action_to_step = arr.item()
                elif arr.ndim == 1:
                    if arr.size == 1:
                        action_to_step = arr.item()
                    else:
                        action_to_step = arr
                else:
                    first = arr[0]
                    if np.asarray(first).ndim == 0:
                        action_to_step = first.item()
                    else:
                        action_to_step = np.asarray(first)

                observation, reward, terminated, truncated, info = Render_env.step(action_to_step)
                obs = [np.expand_dims(observation, axis=0)]
                episode_rew += reward
                eplen += 1
            print("episod reward :", episode_rew, "episod len :", eplen)
            total_rewards.append(episode_rew)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"reward : {avg_reward} +- {std_reward}(std)")
    return avg_reward, std_reward
