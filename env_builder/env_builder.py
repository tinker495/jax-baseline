import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_builder.seeding import seed_env
from jax_baselines.core.env_protocols import Env, EnvInfo, VectorizedEnv

__all__ = [
    "Env",
    "EnvInfo",
    "VectorizedEnv",
    "EnvPoolVectorizedEnv",
    "GymVectorizedEnv",
    "get_env_builder",
]


def get_env_builder(env_name, env_backend="gymnasium", **kwargs):
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

            env_type, env_id = get_env_type(env_name)
            if env_type == "atari_env":
                env = make_wrap_atari(env_name, clip_rewards=True)
            else:
                env = gym.make(env_name, render_mode=render_mode)
            seed_env(env, seed)
            return env

    env_type = "SingleEnv"
    env_info = {
        "env_type": env_type,
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

    def __init__(self, env_id, worker_num=8, render=False, seed=None):
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
        self.env_info: EnvInfo = {
            "observation_space": observation_space,
            "action_space": self.env.action_space,
            "env_type": env_type,
            "env_id": env_id,
        }

        # Set up action conversion for discrete action spaces
        if not isinstance(self.env.action_space, spaces.Box):
            self.action_conv = lambda a: np.asarray(a).flatten().astype(np.int32)
        else:
            self.action_conv = lambda a: np.asarray(a)

        # Initialize environment
        raw_obs, _ = self.env.reset()
        self.obs = self._process_observations(raw_obs)

        # Storage for step/get_result pattern compatibility
        self._pending_result = None

    def _check_atari_env(self, env_id: str) -> bool:
        """Check if the environment is an Atari game."""
        import envpool

        try:
            spec = envpool.make_spec(env_id)
            # EnvPool Atari envs have specific attributes
            return hasattr(spec, "stack_num") or "Atari" in str(type(spec))
        except (KeyError, ValueError):
            pass

        # Fallback: check common Atari game names
        atari_games = [
            "Pong",
            "Breakout",
            "SpaceInvaders",
            "Qbert",
            "Seaquest",
            "BeamRider",
            "Enduro",
            "Asterix",
            "MsPacman",
            "Freeway",
            "Assault",
            "Alien",
            "BankHeist",
            "BattleZone",
            "Boxing",
            "Centipede",
            "DemonAttack",
            "DoubleDunk",
            "Frostbite",
            "Gopher",
            "Hero",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "Phoenix",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Skiing",
            "Tennis",
            "TimePilot",
            "UpNDown",
            "Venture",
            "WizardOfWor",
            "Zaxxon",
            "Amidar",
            "Atlantis",
            "CrazyClimber",
            "Gravitar",
            "JamesBond",
            "MontezumaRevenge",
            "PrivateEye",
            "Solaris",
        ]
        return any(game.lower() in env_id.lower() for game in atari_games)

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def step(self, actions):
        """Execute actions in all environments.

        This method stores the step for get_result() to maintain
        compatibility with the existing async step/get_result pattern.
        """
        # Convert actions to appropriate format
        actions = self.action_conv(actions)

        # Execute step (EnvPool is synchronous but we buffer for compatibility)
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        obs = self._process_observations(obs)

        # Store results for get_result()
        self._pending_result = (obs, rewards, terminateds, truncateds, infos)

    def get_result(self):
        """Get the results of the previous step.

        Returns:
            next_obs: Next observations (num_envs, ...)
            rewards: Rewards (num_envs,)
            terminateds: Terminated flags (num_envs,)
            truncateds: Truncated flags (num_envs,)
            infos: Info dicts
        """
        if self._pending_result is None:
            raise RuntimeError("get_result() called without a preceding step()")

        next_obs, rewards, terminateds, truncateds, infos = self._pending_result
        self._pending_result = None

        if self._is_atari and isinstance(infos, dict) and "original_reward" not in infos:
            original_reward = infos.get("reward")
            if original_reward is not None:
                infos = dict(infos)
                infos["original_reward"] = original_reward

        # EnvPool handles auto-reset internally
        # After done, next_obs already contains the new episode's observation
        # We just need to update our current obs tracker
        self.obs = next_obs

        return next_obs, rewards, terminateds, truncateds, infos

    def close(self):
        """Close the environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def _process_observations(self, obs: np.ndarray) -> np.ndarray:
        """Convert EnvPool outputs to channel-last format expected by models."""
        if self._is_atari and obs.ndim == 4:
            # EnvPool Atari obs format: (N, C, H, W) -> convert to (N, H, W, C)
            return np.transpose(obs, (0, 2, 3, 1))
        return obs

    def _format_observation_space(self, obs_space):
        """Return observation space matching processed observation format."""
        if self._is_atari:
            low = np.transpose(obs_space.low, (1, 2, 0))
            high = np.transpose(obs_space.high, (1, 2, 0))
            return spaces.Box(low=low, high=high, dtype=obs_space.dtype)
        return obs_space


class GymVectorizedEnv(VectorizedEnv):
    """Fallback vectorized environment using gymnasium's native vectorization.

    Used when EnvPool doesn't support the requested environment.
    """

    def __init__(self, env_id, worker_num=8, render=False, seed=None):
        self.env_id = env_id
        self.worker_num = worker_num

        # Create vectorized environment using gymnasium
        # For Atari, we need to use custom wrappers, so we use AsyncVectorEnv with explicit constructors
        from env_builder.atari_wrappers import get_env_type, make_wrap_atari

        env_type, _ = get_env_type(env_id)

        def make_env(seed_offset=0):
            def _make():
                if env_type == "atari_env":
                    env = make_wrap_atari(env_id, clip_rewards=True)
                else:
                    env = gym.make(env_id)

                if seed is not None:
                    seed_env(env, seed + seed_offset)
                return env

            return _make

        if hasattr(gym, "make_vec") and env_type != "atari_env":
            # For non-Atari, try efficient make_vec if available
            try:
                self.env = gym.make_vec(env_id, num_envs=worker_num, vectorization_mode="async")
            except Exception:
                self.env = gym.vector.AsyncVectorEnv([make_env(i) for i in range(worker_num)])
        elif hasattr(gym.vector, "AsyncVectorEnv"):
            # For Atari or if make_vec fails/not avail, use AsyncVectorEnv manually to ensure wrappers applied
            self.env = gym.vector.AsyncVectorEnv([make_env(i) for i in range(worker_num)])
        else:
            # Last resort fallback
            self.env = gym.vector.make(env_id, num_envs=worker_num, asynchronous=True)

        # Store environment info
        self.env_info: EnvInfo = {
            "observation_space": self.env.single_observation_space,
            "action_space": self.env.single_action_space,
            "env_type": "gym_vector",
            "env_id": env_id,
        }

        # Set up action conversion
        if not isinstance(self.env.single_action_space, spaces.Box):
            self.action_conv = lambda a: np.asarray(a).flatten().astype(np.int32)
        else:
            self.action_conv = lambda a: np.asarray(a)

        # Initialize
        self.obs, _ = self.env.reset()
        self._pending_result = None

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def step(self, actions):
        actions = self.action_conv(actions)
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        self._pending_result = (obs, rewards, terminateds, truncateds, infos)

    def get_result(self):
        if self._pending_result is None:
            raise RuntimeError("get_result() called without a preceding step()")

        next_obs, rewards, terminateds, truncateds, infos = self._pending_result
        self._pending_result = None
        self.obs = next_obs

        return next_obs, rewards, terminateds, truncateds, infos

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
