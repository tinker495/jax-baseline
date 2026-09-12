import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.core.env_protocols import (
    EnvInfo,
    EvaluationContextEnv,
    PreparedEnvSpec,
    PreparedWorkerEnvSpec,
    SingleEnv,
    VectorizedEnv,
    VectorizedEvalEnv,
)

REQUIRED_ENV_INFO_KEYS = (
    "observation_space",
    "action_size",
    "action_type",
    "env_type",
    "env_id",
    "worker_num",
    "core_env_type",
    "runtime",
)


# Action-space converters (defined at module scope to avoid lambda-based E731 lint issues)
def _discrete_action_conv(a):
    return a[0]


def _continuous_action_conv(a):
    if isinstance(a, jax.Array):
        return jnp.clip(a, -5.0, 5.0)
    return np.clip(a, -5.0, 5.0)


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

    runtime = env_info["runtime"]
    if not isinstance(runtime, dict):
        raise TypeError("Prepared env_info runtime must be a dict")
    for key in ("backend", "backend_env_id", "seed_rule", "reward_clipping"):
        if key not in runtime or not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"Prepared env_info runtime {key} must be a non-empty string")
    if "seed" not in runtime or (runtime["seed"] is not None and type(runtime["seed"]) is not int):
        raise ValueError("Prepared env_info runtime seed must be an int or None")
    if "episodic_life" not in runtime or not isinstance(runtime["episodic_life"], bool):
        raise ValueError("Prepared env_info runtime episodic_life must be a bool")

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
    if prepared.env is prepared.eval_env and not isinstance(prepared.env, EvaluationContextEnv):
        raise ValueError("Shared train/eval env must provide an evaluation_context()")
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

    if env_type == "VectorizedEnv":
        if not isinstance(prepared.env, VectorizedEnv):
            raise ValueError(
                "Prepared train env metadata says VectorizedEnv but env is not VectorizedEnv"
            )
        if not isinstance(prepared.eval_env, VectorizedEvalEnv):
            raise ValueError("Prepared eval env must satisfy the VectorizedEvalEnv protocol")
        eval_info = _require_env_info(prepared.eval_env.get_info())
        if _validate_core_env_type(eval_info) != env_type or eval_info["worker_num"] != worker_size:
            raise ValueError(
                "Prepared train and eval envs must have the same type and worker count"
            )
    else:
        if (
            worker_size != 1
            or isinstance(prepared.env, VectorizedEnv)
            or isinstance(prepared.eval_env, VectorizedEnv)
        ):
            raise ValueError(
                "Prepared single train and eval envs must be single-worker environments"
            )
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
    if not isinstance(workers, list):
        raise ValueError("Invalid workers type")
    env_info = _require_env_info(worker_info(workers[0]))
    observation_space = env_info["observation_space"]
    action_size = env_info["action_size"]
    action_type = env_info["action_type"]
    env_type = _validate_core_env_type(env_info)

    if include_action_type:
        return observation_space, action_size, env_type, action_type
    return observation_space, action_size, env_type


def infer_action_meta(action_type):
    """Return (action_type, conv_action) for adapter-normalized action metadata."""
    if action_type == "discrete":
        return action_type, _discrete_action_conv
    if action_type == "continuous":
        return action_type, _continuous_action_conv
    raise ValueError(f"Unsupported action type: {action_type!r}")
