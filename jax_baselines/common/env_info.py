import numpy as np

from jax_baselines.common.env_protocols import EnvInfo, SingleEnv, VectorizedEnv

REQUIRED_ENV_INFO_KEYS = ("observation_space", "action_space", "env_type", "env_id")


# Action-space converters (defined at module scope to avoid lambda-based E731 lint issues)
def _discrete_action_conv(a):
    return a[0]


def _continuous_action_conv(a):
    return np.clip(a, -3.0, 3.0) / 3.0


def _observation_space_shape(observation_space):
    if hasattr(observation_space, "shape"):
        return [list(observation_space.shape)]
    return [list(observation_space)]


def _is_discrete_space(action_space):
    return hasattr(action_space, "n")


def _action_size(action_space):
    if _is_discrete_space(action_space):
        return [action_space.n]
    if hasattr(action_space, "shape") and action_space.shape:
        return [action_space.shape[0]]
    raise ValueError(f"Unsupported action space type: {type(action_space)}")


def _is_single_env(env):
    return isinstance(env, SingleEnv) or (
        hasattr(env, "observation_space")
        and hasattr(env, "action_space")
        and callable(getattr(env, "reset", None))
        and callable(getattr(env, "step", None))
    )


def _require_env_info(env_info: EnvInfo | None) -> EnvInfo:
    if env_info is None:
        raise ValueError("VectorizedEnv env_info is required")

    missing = [key for key in REQUIRED_ENV_INFO_KEYS if key not in env_info]
    if missing:
        raise ValueError(f"VectorizedEnv env_info missing required keys: {', '.join(missing)}")

    return env_info


def get_local_env_info(env_builder, num_workers=1, seed=None):
    """Create envs and extract standardized info used by base classes.

    Returns: (env, eval_env, observation_space, action_size, worker_size, env_type)
    """
    eval_seed = None if seed is None else seed + 1
    env = env_builder(num_workers, seed=seed)
    eval_env = env_builder(1, seed=eval_seed)

    if isinstance(env, VectorizedEnv):
        env_info = _require_env_info(env.env_info)
        obs_space = env_info["observation_space"]
        act_space = env_info["action_space"]

        observation_space = _observation_space_shape(obs_space)
        action_size = _action_size(act_space)
        worker_size = env.worker_num
        env_type = "VectorizedEnv"
    elif _is_single_env(env):
        action_space = env.action_space
        observation_space = _observation_space_shape(env.observation_space)
        action_size = _action_size(action_space)
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

    if isinstance(workers, list):
        env_dict = ray.get(workers[0].get_info.remote())
        observation_space = [list(env_dict["observation_space"].shape)]
        action_space = env_dict["action_space"]

        if _is_discrete_space(action_space):
            action_size = [action_space.n]
            action_type = "discrete"
        else:
            action_size = _action_size(action_space)
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
    if _is_discrete_space(action_space):
        action_type = "discrete"
        conv_action = _discrete_action_conv
    else:
        _action_size(action_space)
        action_type = "continuous"
        conv_action = _continuous_action_conv
    return action_type, conv_action
