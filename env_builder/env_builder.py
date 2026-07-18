import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.utils import rescale_box

from env_builder.observations import (
    flatten_observation_space,
    normalize_observation,
    normalize_observation_space,
)
from env_builder.seeding import seed_env
from jax_baselines.core.env_protocols import (
    Env,
    EnvInfo,
    PreparedEnvSpec,
    PreparedWorkerEnvSpec,
    SingleEnv,
    VectorizedEnv,
)

__all__ = [
    "Env",
    "EnvInfo",
    "PreparedEnvSpec",
    "PreparedWorkerEnvSpec",
    "VectorizedEnv",
    "EnvPoolVectorizedEnv",
    "GymVectorizedEnv",
    "get_env_builder",
]


def _action_meta(action_space) -> tuple[list[int], str]:
    if hasattr(action_space, "n"):
        return [int(action_space.n)], "discrete"
    if hasattr(action_space, "shape") and action_space.shape:
        return [int(action_space.shape[0])], "continuous"
    raise ValueError(f"Unsupported action space type: {type(action_space)}")


def _normalize_action_space(env):
    action_space = env.action_space
    if not isinstance(action_space, spaces.Box):
        return env
    if not (np.isfinite(action_space.low).all() and np.isfinite(action_space.high).all()):
        return env
    unit = np.ones(action_space.shape, dtype=action_space.dtype)
    return gym.wrappers.RescaleAction(env, min_action=-unit, max_action=unit)


def _real_reset_mask(is_atari, terminateds, truncateds, infos):
    terminateds = np.asarray(terminateds, dtype=bool)
    truncateds = np.asarray(truncateds, dtype=bool)
    if not (is_atari and isinstance(infos, dict) and "lives" in infos):
        return terminateds | truncateds
    lives = np.asarray(infos["lives"], dtype=np.int32)
    if lives.shape == ():
        lives = np.full(terminateds.shape, lives.item(), dtype=np.int32)
    return truncateds | (terminateds & (lives.reshape(-1) == 0))


def _autoreset_mask(terminateds, truncateds):
    return np.asarray(terminateds, dtype=bool) | np.asarray(truncateds, dtype=bool)


def _single_env_info(env, env_id: str) -> EnvInfo:
    if not isinstance(env, SingleEnv):
        raise ValueError("Single env must satisfy the SingleEnv protocol")
    action_size, action_type = _action_meta(env.action_space)
    return {
        "observation_space": normalize_observation_space(env.observation_space),
        "action_size": action_size,
        "action_type": action_type,
        "env_type": "single",
        "env_id": env_id,
        "worker_num": 1,
        "core_env_type": "SingleEnv",
    }


def _prepared_env_info(env, env_id: str) -> EnvInfo:
    if isinstance(env, VectorizedEnv):
        env_info = env.env_info
        if env_info is None:
            raise ValueError("Vectorized env must expose env_info")
        return env_info
    return _single_env_info(env, env_id)


def get_env_builder(env_name, env_backend="gymnasium"):
    if env_backend not in ("gymnasium", "envpool"):
        raise ValueError(f"env_backend must be 'gymnasium' or 'envpool', got {env_backend!r}")

    def env_builder(worker=1, render_mode=None, seed=None):
        if worker > 1:
            # Vectorized backend is an explicit choice: gymnasium AsyncVectorEnv
            # (default, portable) or EnvPool (faster, only for envs it ships).
            if env_backend == "envpool":
                if not _is_envpool_supported(env_name):
                    raise ValueError(
                        f"env_backend='envpool' requested but EnvPool has no spec for "
                        f"{env_name!r}; use env_backend='gymnasium' or a supported env id."
                    )
                return EnvPoolVectorizedEnv(env_name, worker_num=worker, seed=seed)
            return GymVectorizedEnv(env_name, worker_num=worker, seed=seed)
        else:
            from env_builder.atari_wrappers import get_env_type, make_wrap_atari

            env_type, _ = get_env_type(env_name)
            if env_type == "atari_env":
                env = make_wrap_atari(env_name, clip_rewards=True)
            else:
                env = gym.make(env_name, render_mode=render_mode)
            env = gym.wrappers.TransformObservation(
                env,
                normalize_observation,
                spaces.Dict(flatten_observation_space(env.observation_space)),
            )
            env = _normalize_action_space(env)
            seed_env(env, seed)
            return env

    def prepare_envs(num_workers=1, seed=None):
        eval_seed = None if seed is None else seed + 1
        env = env_builder(num_workers, seed=seed)
        eval_env = env_builder(1, seed=eval_seed)
        return PreparedEnvSpec(
            env=env,
            eval_env=eval_env,
            env_info=_prepared_env_info(env, env_name),
        )

    def prepare_worker_env(seed=None):
        env = env_builder(1, seed=seed)
        return PreparedWorkerEnvSpec(env=env, env_info=_single_env_info(env, env_name))

    env_builder.prepare_envs = prepare_envs
    env_builder.prepare_worker_env = prepare_worker_env

    env_info = {
        "env_type": "adapter_factory",
        "env_id": env_name,
    }
    return env_builder, env_info


def _get_envpool_env_id(env_name: str) -> str:
    """Convert environment name to EnvPool compatible format.

    EnvPool uses different naming conventions:
    - Atari: "Pong-v5" instead of "ALE/Pong-v5" or "PongNoFrameskip-v4"
    - MuJoCo: Same as gymnasium (e.g., "HalfCheetah-v4")
    - Classic: Same as gymnasium (e.g., "CartPole-v1")
    """
    # Handle ALE/ prefix
    if env_name.startswith("ALE/"):
        env_name = env_name[4:]

    # Handle NoFrameskip Atari environments
    if "NoFrameskip" in env_name:
        # Convert "PongNoFrameskip-v4" to "Pong-v5"
        return env_name.replace("NoFrameskip", "").replace("-v4", "-v5")

    return env_name


def _is_envpool_supported(env_name: str) -> bool:
    """Return True if EnvPool has a spec for ``env_name``.

    A missing envpool install is surfaced (it silently disables the fast path);
    an unknown env id is the expected "not supported" outcome and stays quiet.
    """
    try:
        import envpool
    except ImportError:
        warnings.warn(
            "envpool is not installed; vectorized envs use the slower gymnasium "
            "AsyncVectorEnv. Install envpool to enable the fast path.",
            stacklevel=2,
        )
        return False
    try:
        envpool.make_spec(_get_envpool_env_id(env_name))
        return True
    except (KeyError, ValueError):
        return False


class EnvPoolVectorizedEnv(VectorizedEnv):
    """High-performance vectorized environment using EnvPool.

    EnvPool provides C++ based parallel environment execution,
    achieving much higher throughput than Ray-based parallelization.

    Features:
    - Auto-reset: Environments automatically reset when done
    - Synchronous API: Compatible with existing training loops
    - High performance: Up to 1M FPS for Atari, 3M FPS for MuJoCo
    """

    def __init__(self, env_id, worker_num=8, seed=None):
        import envpool

        self.env_id = env_id
        self.worker_num = worker_num

        # Convert env_id to EnvPool format
        envpool_env_id = _get_envpool_env_id(env_id)

        # Determine if this is an Atari environment
        self._is_atari = self._check_atari_env(envpool_env_id)

        # Create EnvPool environment
        # EnvPool uses 'gymnasium' env_type for gymnasium compatibility
        env_kwargs = {
            "env_type": "gymnasium",
            "num_envs": worker_num,
            # Lockstep async: with batch_size == num_envs every recv() waits for
            # all N envs, preserving the fixed-N-transitions-per-step contract
            # the algorithms rely on while still overlapping env stepping with
            # the caller's work between step() and get_result().
            "batch_size": worker_num,
        }

        if seed is not None:
            env_kwargs["seed"] = seed

        # Atari-specific settings (matching existing atari_wrappers behavior)
        if self._is_atari:
            env_kwargs.update(
                {
                    "stack_num": 4,  # Frame stacking
                    "frame_skip": 4,  # Frame skipping
                    "episodic_life": True,  # Episodic life
                    "reward_clip": True,  # Clip rewards to {-1, 0, 1}
                    "img_height": 84,
                    "img_width": 84,
                    "gray_scale": True,
                }
            )

        self.env = envpool.make(envpool_env_id, **env_kwargs)

        # Determine environment type for compatibility
        env_type = "atari_env" if self._is_atari else "envpool"

        # Store environment info matching the existing interface
        observation_space = self._format_observation_space(self.env.observation_space)
        action_size, action_type = _action_meta(self.env.action_space)
        self.env_info: EnvInfo = {
            "observation_space": observation_space,
            "action_size": action_size,
            "action_type": action_type,
            "env_type": env_type,
            "env_id": env_id,
            "worker_num": worker_num,
            "core_env_type": "VectorizedEnv",
        }

        # Set up action conversion for the normalized [-1, 1] core contract.
        if not isinstance(self.env.action_space, spaces.Box):
            self.action_conv = lambda a: np.asarray(a).flatten().astype(np.int32)
        elif (
            np.isfinite(self.env.action_space.low).all()
            and np.isfinite(self.env.action_space.high).all()
        ):
            unit = np.ones(self.env.action_space.shape, dtype=self.env.action_space.dtype)
            _, _, self.action_conv = rescale_box(self.env.action_space, -unit, unit)
        else:
            self.action_conv = lambda a: np.asarray(a)

        # env_id vector for send(); recv() may hand back envs in completion
        # order, so every result is re-sorted to the canonical 0..N-1 layout
        # the training loop indexes by (scores, prev_done, replay rows).
        self._all_env_ids = np.arange(worker_num, dtype=np.int32)
        self._awaiting_recv = False

        # Async handshake: async_reset() launches every reset on the C++ side,
        # the first recv() collects the initial observation. From here the
        # contract is a strict send (step) / recv (get_result) alternation.
        self.env.async_reset()
        raw_obs, _, _, _, info = self.env.recv()
        self.obs = self._process_observations(raw_obs, np.argsort(info["env_id"]))

    def _check_atari_env(self, env_id: str) -> bool:
        """Check if the environment is an Atari game."""
        import envpool

        spec = envpool.make_spec(env_id)
        return "Atari" in type(spec).__name__

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def step(self, actions):
        """Fire actions into all environments without blocking.

        ``send()`` hands the actions to EnvPool's C++ worker threads and
        returns immediately, so the caller can do useful work (e.g. a gradient
        step) while the environments advance. ``get_result()`` collects the
        outcome via ``recv()``.
        """
        self.env.send(self.action_conv(actions), self._all_env_ids)
        self._awaiting_recv = True

    def get_result(self):
        """Block until the in-flight async step finishes and return its result.

        Returns:
            next_obs: Next observations (num_envs, ...)
            rewards: Rewards (num_envs,)
            terminateds: Terminated flags (num_envs,)
            truncateds: Truncated flags (num_envs,)
            infos: Info dicts
        """
        if not self._awaiting_recv:
            raise RuntimeError("get_result() called without a preceding step()")
        self._awaiting_recv = False

        next_obs, rewards, terminateds, truncateds, infos = self.env.recv()
        order = np.argsort(infos["env_id"])
        next_obs = self._process_observations(next_obs, order)
        rewards = rewards[order]
        terminateds = terminateds[order]
        truncateds = truncateds[order]
        infos = self._reorder_info(infos, order)

        if self._is_atari and "original_reward" not in infos:
            original_reward = infos.get("reward")
            if original_reward is not None:
                infos["original_reward"] = original_reward

        # EnvPool handles auto-reset internally: after a done flag, next_obs
        # already holds the new episode's first observation.
        self.obs = next_obs

        return next_obs, rewards, terminateds, truncateds, infos

    def real_reset_mask(self, terminateds, truncateds, infos):
        return _real_reset_mask(self._is_atari, terminateds, truncateds, infos)

    def autoreset_mask(self, terminateds, truncateds, infos):
        return _real_reset_mask(self._is_atari, terminateds, truncateds, infos)

    def _reorder_info(self, infos, order):
        """Re-sort per-env info arrays into the canonical 0..N-1 layout.

        Only top-level arrays whose leading axis is the worker dimension are
        permuted -- that covers every field the algorithms consume (``reward``,
        ``lives``, their presence masks, ``env_id``). Nested values such as
        EnvPool's ``players`` dict are passed through unchanged: nothing reads
        them, and under the lockstep ``batch_size == num_envs`` config ``recv()``
        empirically returns envs already in order, so ``order`` is the identity
        and the reorder is purely defensive.
        """
        return {
            key: value[order]
            if isinstance(value, np.ndarray) and value.shape and value.shape[0] == self.worker_num
            else value
            for key, value in infos.items()
        }

    def close(self):
        """Close the environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def _process_observations(self, obs, order):
        """Convert EnvPool outputs to channel-last format expected by models."""
        normalized = {key: value[order] for key, value in normalize_observation(obs).items()}
        if self._is_atari:
            normalized = {
                key: np.transpose(value, (0, 2, 3, 1)) if value.ndim == 4 else value
                for key, value in normalized.items()
            }
        return normalized

    def _format_observation_space(self, obs_space):
        """Return observation space matching processed observation format."""
        if self._is_atari:
            low = np.transpose(obs_space.low, (1, 2, 0))
            high = np.transpose(obs_space.high, (1, 2, 0))
            obs_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)
        return normalize_observation_space(obs_space)


class GymVectorizedEnv(VectorizedEnv):
    """Fallback vectorized environment using gymnasium's native vectorization.

    Used when EnvPool doesn't support the requested environment.
    """

    def __init__(self, env_id, worker_num=8, seed=None):
        self.env_id = env_id
        self.worker_num = worker_num

        # Create vectorized environment using gymnasium
        # For Atari, we need to use custom wrappers, so we use AsyncVectorEnv with explicit constructors
        from env_builder.atari_wrappers import get_env_type, make_wrap_atari

        env_type, _ = get_env_type(env_id)
        self._is_atari = env_type == "atari_env"

        def make_env():
            def _make():
                if self._is_atari:
                    env = make_wrap_atari(env_id, clip_rewards=True)
                else:
                    env = gym.make(env_id)
                return _normalize_action_space(env)

            return _make

        if env_type != "atari_env":
            # Non-Atari: prefer the registry's efficient make_vec, falling back to
            # explicit AsyncVectorEnv if the env has no vectorized entry point.
            try:
                self.env = gym.make_vec(
                    env_id,
                    num_envs=worker_num,
                    vectorization_mode="async",
                    vector_kwargs={"context": "spawn"},
                    wrappers=(_normalize_action_space,),
                )
            except Exception:
                self.env = gym.vector.AsyncVectorEnv(
                    [make_env() for _ in range(worker_num)], context="spawn"
                )
        else:
            # Atari needs the custom wrappers, so build AsyncVectorEnv from the
            # explicit per-env constructors.
            self.env = gym.vector.AsyncVectorEnv(
                [make_env() for _ in range(worker_num)], context="spawn"
            )

        # Store environment info
        action_size, action_type = _action_meta(self.env.single_action_space)
        self.env_info: EnvInfo = {
            "observation_space": normalize_observation_space(self.env.single_observation_space),
            "action_size": action_size,
            "action_type": action_type,
            "env_type": "gym_vector",
            "env_id": env_id,
            "worker_num": worker_num,
            "core_env_type": "VectorizedEnv",
        }

        # Set up action conversion
        if not isinstance(self.env.single_action_space, spaces.Box):
            self.action_conv = lambda a: np.asarray(a).flatten().astype(np.int32)
        else:
            self.action_conv = lambda a: np.asarray(a)

        # gymnasium vector envs split into step_async()/step_wait(); for
        # AsyncVectorEnv each sub-env runs in its own subprocess, so
        # step_async() returns immediately and the envs advance while the
        # caller works, with step_wait() collecting. Every construction path
        # above builds an async-capable vector env; fail loudly if one without
        # the split slips through (e.g. SyncVectorEnv lacks it).
        if not (hasattr(self.env, "step_async") and hasattr(self.env, "step_wait")):
            raise TypeError(
                f"{type(self.env).__name__} lacks step_async/step_wait; "
                "GymVectorizedEnv requires an async-capable vector env."
            )

        # Initialize
        self.obs, _ = self.env.reset(seed=seed)
        self.obs = normalize_observation(self.obs)
        self._awaiting_result = False

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def step(self, actions):
        """Dispatch actions without blocking (gymnasium ``step_async``)."""
        self.env.step_async(self.action_conv(actions))
        self._awaiting_result = True

    def get_result(self):
        """Block on the in-flight async step and return its result."""
        if not self._awaiting_result:
            raise RuntimeError("get_result() called without a preceding step()")
        self._awaiting_result = False

        next_obs, rewards, terminateds, truncateds, infos = self.env.step_wait()
        next_obs = normalize_observation(next_obs)
        self.obs = next_obs

        return next_obs, rewards, terminateds, truncateds, infos

    def real_reset_mask(self, terminateds, truncateds, infos):
        return _real_reset_mask(self._is_atari, terminateds, truncateds, infos)

    def autoreset_mask(self, terminateds, truncateds, infos):
        return _autoreset_mask(terminateds, truncateds)

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
