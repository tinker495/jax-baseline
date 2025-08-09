import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jax_baselines.common.env_builder import VectorizedEnv


# Action-space converters (defined at module scope to avoid lambda-based E731 lint issues)
def _discrete_action_conv(a):
    return a[0]


def _continuous_action_conv(a):
    return np.clip(a, -3.0, 3.0) / 3.0


def get_local_env_info(env_builder, num_workers=1):
    """Create envs and extract standardized info used by base classes.

    Returns: (env, eval_env, observation_space, action_size, worker_size, env_type)
    """
    env = env_builder(num_workers)
    eval_env = env_builder(1)

    if isinstance(env, VectorizedEnv):
        env_info = env.env_info
        observation_space = [list(env_info["observation_space"].shape)]
        action_size = [
            env_info["action_space"].n
            if not isinstance(env_info["action_space"], spaces.Box)
            else env_info["action_space"].shape[0]
        ]
        worker_size = env.worker_num
        env_type = "VectorizedEnv"
    elif isinstance(env, gym.Env) or isinstance(env, gym.Wrapper):
        action_space = env.action_space
        observation_space = [list(env.observation_space.shape)]
        action_size = [
            action_space.n if not isinstance(action_space, spaces.Box) else action_space.shape[0]
        ]
        worker_size = 1
        env_type = "SingleEnv"
    else:
        raise ValueError("Unsupported env type")

    return env, eval_env, observation_space, action_size, worker_size, env_type


def get_remote_env_info(workers, include_action_type=False):
    """Get standardized environment info from remote workers (ray actors).

    Args:
        workers: List of ray actors with get_info.remote() method
        include_action_type: If True, also return action_type

    Returns:
        observation_space, action_size, env_type [, action_type]
    """
    import ray
    from gymnasium import spaces

    if isinstance(workers, list):
        env_dict = ray.get(workers[0].get_info.remote())
        observation_space = [list(env_dict["observation_space"].shape)]

        if not isinstance(env_dict["action_space"], spaces.Box):
            action_size = [env_dict["action_space"].n]
            action_type = "discrete"
        else:
            action_size = [env_dict["action_space"].shape[0]]
            action_type = "continuous"

        env_type = "SingleEnv"

        if include_action_type:
            return observation_space, action_size, env_type, action_type
        else:
            return observation_space, action_size, env_type
    else:
        raise ValueError("Invalid workers type")


def infer_action_meta(action_space):
    """Return (action_type, conv_action) for given gym action_space."""
    if not isinstance(action_space, spaces.Box):
        action_type = "discrete"
        conv_action = _discrete_action_conv
    else:
        action_type = "continuous"
        conv_action = _continuous_action_conv
    return action_type, conv_action
