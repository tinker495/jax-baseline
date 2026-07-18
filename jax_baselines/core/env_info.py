import numpy as np

from jax_baselines.core.env_protocols import (
    EnvInfo,
    PreparedEnvSpec,
    PreparedWorkerEnvSpec,
    SingleEnv,
    VectorizedEnv,
)

REQUIRED_ENV_INFO_KEYS = (
    "observation_space",
    "action_size",
    "action_type",
    "env_type",
    "env_id",
    "worker_num",
    "core_env_type",
)


# Action-space converters (defined at module scope to avoid lambda-based E731 lint issues)
def _discrete_action_conv(a):
    return a[0]


def _continuous_action_conv(a):
    return np.clip(a, -3.0, 3.0) / 3.0


def _require_env_info(env_info: EnvInfo | None) -> EnvInfo:
    if env_info is None:
        raise ValueError("Prepared env_info is required")

    missing = [key for key in REQUIRED_ENV_INFO_KEYS if key not in env_info]
    if missing:
        raise ValueError(f"Prepared env_info missing required keys: {', '.join(missing)}")

    observation_space = env_info["observation_space"]
    if not isinstance(observation_space, dict) or not observation_space:
        raise ValueError("Prepared env_info observation_space must be a non-empty dict")
    if any(not isinstance(key, str) for key in observation_space):
        raise ValueError("Prepared env_info observation keys must be strings")

    return env_info


def _require_single_env(env, context: str):
    if not isinstance(env, SingleEnv):
        raise ValueError(f"{context} must satisfy the SingleEnv protocol")


def _validate_core_env_type(env_info: EnvInfo) -> str:
    env_type = env_info["core_env_type"]
    if env_type not in {"SingleEnv", "VectorizedEnv"}:
        raise ValueError(f"Unsupported core_env_type: {env_type!r}")
    return env_type


def _prepare_envs(env_builder, num_workers=1, seed=None):
    prepare = getattr(env_builder, "prepare_envs", None)
    if not callable(prepare):
        raise ValueError("Environment adapter must expose prepare_envs(num_workers=..., seed=...)")
    prepared = prepare(num_workers=num_workers, seed=seed)
    if not isinstance(prepared, PreparedEnvSpec):
        raise ValueError("prepare_envs must return PreparedEnvSpec")
    if prepared.eval_env is None:
        raise ValueError("Prepared local eval_env is required")
    return prepared


def get_local_env_info(env_builder, num_workers=1, seed=None, include_action_type=False):
    """Extract standardized info from adapter-prepared train/eval envs.

    Returns: (env, eval_env, observation_space, action_size, worker_size, env_type)
    """
    prepared = _prepare_envs(env_builder, num_workers=num_workers, seed=seed)
    env_info = _require_env_info(prepared.env_info)

    observation_space = env_info["observation_space"]
    action_size = env_info["action_size"]
    worker_size = int(env_info["worker_num"])
    env_type = _validate_core_env_type(env_info)

    if env_type == "VectorizedEnv" and not isinstance(prepared.env, VectorizedEnv):
        raise ValueError(
            "Prepared train env metadata says VectorizedEnv but env is not VectorizedEnv"
        )
    if env_type == "SingleEnv":
        _require_single_env(prepared.env, "Prepared train env")
    _require_single_env(prepared.eval_env, "Prepared eval env")

    result = (
        prepared.env,
        prepared.eval_env,
        observation_space,
        action_size,
        worker_size,
        env_type,
    )
    if include_action_type:
        return (*result, env_info["action_type"])
    return result


def prepare_worker_env(env_builder, seed=None):
    """Return a single worker env and adapter-provided normalized metadata."""
    prepare = getattr(env_builder, "prepare_worker_env", None)
    if not callable(prepare):
        raise ValueError("Environment adapter must expose prepare_worker_env(seed=...)")
    prepared = prepare(seed=seed)
    if not isinstance(prepared, PreparedWorkerEnvSpec):
        raise ValueError("prepare_worker_env must return PreparedWorkerEnvSpec")
    env_info = _require_env_info(prepared.env_info)
    if _validate_core_env_type(env_info) != "SingleEnv":
        raise ValueError("Prepared worker env metadata must be SingleEnv")
    if int(env_info["worker_num"]) != 1:
        raise ValueError("Prepared worker env worker_num must be 1")
    _require_single_env(prepared.env, "Prepared worker env")
    return prepared.env, env_info


def get_worker_env_info(workers, worker_info, include_action_type=False):
    """Get standardized environment info from distributed worker handles.

    Args:
        workers: List of worker handles with normal get_info semantics.
        worker_info: Runtime adapter callable that returns one worker's info.
        include_action_type: If True, also return action_type

    Returns:
        observation_space, action_size, env_type [, action_type]
    """
    if isinstance(workers, list):
        env_dict = worker_info(workers[0])
        env_info = _require_env_info(env_dict)
        observation_space = env_info["observation_space"]
        action_size = env_info["action_size"]
        action_type = env_info["action_type"]
        env_type = _validate_core_env_type(env_info)

        if include_action_type:
            return observation_space, action_size, env_type, action_type
        else:
            return observation_space, action_size, env_type
    else:
        raise ValueError("Invalid workers type")


def infer_action_meta(action_type):
    """Return (action_type, conv_action) for adapter-normalized action metadata."""
    if action_type == "discrete":
        return action_type, _discrete_action_conv
    if action_type == "continuous":
        return action_type, _continuous_action_conv
    raise ValueError(f"Unsupported action type: {action_type!r}")
