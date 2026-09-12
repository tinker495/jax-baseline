import random
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import concatenate, iterate
from gymnasium.wrappers.utils import rescale_box

from env_builder.metrics import GymEnvMetrics, GymLoggingWrapper
from env_builder.observations import (
    flatten_observation_space,
    normalize_observation,
    normalize_observation_space,
)
from env_builder.seeding import seed_env
from jax_baselines.core.env_protocols import (
    Env,
    EnvInfo,
    EnvironmentMetadata,
    Observation,
    PreparedEnvSpec,
    PreparedWorkerEnvSpec,
    SingleEnv,
    VectorizedEnv,
)
from jax_baselines.core.runtime_adapters import MetricLogger

__all__ = [
    "Env",
    "EnvInfo",
    "EnvPoolVectorizedEnv",
    "GymVectorizedEnv",
    "PreparedEnvSpec",
    "PreparedWorkerEnvSpec",
    "VectorizedEnv",
    "get_env_builder",
    "get_env_info",
]


def _action_meta(action_space) -> tuple[list[int], Literal["discrete", "continuous"]]:
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
    runtime: EnvironmentMetadata = env.runtime
    if not isinstance(env, SingleEnv):
        raise ValueError("Single env must satisfy the SingleEnv protocol")
    action_size, action_type = _action_meta(env.action_space)
    observation_space = env.observation_space
    if isinstance(observation_space, spaces.Dict):
        observation_space = {
            key: list(leaf.shape) for key, leaf in observation_space.spaces.items()
        }
    return {
        "observation_space": observation_space,
        "action_size": action_size,
        "action_type": action_type,
        "env_type": "single",
        "env_id": env_id,
        "worker_num": 1,
        "core_env_type": "SingleEnv",
        "runtime": runtime,
    }


def get_env_info(env, env_id: str) -> EnvInfo:
    """Describe an environment after the adapter has resolved its backend."""
    if not isinstance(env, VectorizedEnv):
        return _single_env_info(env, env_id)
    return env.get_info()


_ENV_BACKENDS = ("gymnasium", "envpool", "mjlab")


def _close_envs(*envs):
    seen = set()
    for env in envs:
        if env is None or id(env) in seen:
            continue
        seen.add(id(env))
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def get_env_builder(
    env_name,
    env_backend="gymnasium",
    *,
    observation_key=None,
    episode_length=None,
    device="cuda:0",
    jax_arrays=False,
    reuse_for_eval=False,
):
    if env_backend not in _ENV_BACKENDS:
        raise ValueError(f"env_backend must be one of {_ENV_BACKENDS}, got {env_backend!r}")
    if jax_arrays and env_backend != "mjlab":
        raise ValueError("jax_arrays requires env_backend='mjlab'")
    if reuse_for_eval and env_backend == "envpool":
        raise ValueError("EnvPool does not support training environment state preservation")

    def env_builder(worker=1, render_mode=None, seed=None):
        if env_backend == "mjlab":
            from env_builder.mjlab_env import make_mjlab_env

            return make_mjlab_env(
                env_name,
                worker_num=worker,
                seed=seed,
                observation_key=observation_key,
                episode_length=episode_length,
                device=device,
                render_mode=render_mode,
                jax_arrays=jax_arrays,
                reuse_for_eval=reuse_for_eval and render_mode is None,
            )
        # Vectorized backend is an explicit choice: gymnasium AsyncVectorEnv
        # (default, portable) or EnvPool (faster, only for envs it ships).
        if worker > 1 and env_backend == "envpool":
            if not _is_envpool_supported(env_name):
                raise ValueError(
                    f"env_backend='envpool' requested but EnvPool has no spec for "
                    f"{env_name!r}; use env_backend='gymnasium' or a supported env id."
                )
            return EnvPoolVectorizedEnv(
                env_name,
                worker_num=worker,
                seed=seed,
                observation_key=observation_key,
            )
        if worker > 1:
            return GymVectorizedEnv(
                env_name,
                worker_num=worker,
                seed=seed,
                observation_key=observation_key,
                reuse_for_eval=reuse_for_eval and render_mode is None,
            )
        from env_builder.atari_wrappers import get_env_type, make_wrap_atari

        env_type, _ = get_env_type(env_name)
        if env_type == "atari_env":
            env = make_wrap_atari(env_name, clip_rewards=True)
        else:
            env = gym.make(env_name, render_mode=render_mode)
        env = gym.wrappers.TransformObservation(
            env,
            lambda observation: normalize_observation(observation, observation_key),
            spaces.Dict(flatten_observation_space(env.observation_space, observation_key)),
        )
        env = _normalize_action_space(env)
        if reuse_for_eval and render_mode is None:
            from env_builder.gym_state import GymStateWrapper

            try:
                env = GymStateWrapper(env)
            except (TypeError, ValueError):
                env.close()
                raise
            if seed is None:
                env.reset()
        seed_env(env, seed)
        return GymLoggingWrapper(env, env_id=env_name, seed=seed, is_atari=env_type == "atari_env")

    def prepare_envs(num_workers=1, seed=None):
        eval_seed = None if seed is None else seed + 1
        env = eval_env = None
        try:
            env = env_builder(num_workers, seed=seed)
            eval_env = env if reuse_for_eval else env_builder(num_workers, seed=eval_seed)
            return PreparedEnvSpec(
                env=env,
                eval_env=eval_env,
                env_info=get_env_info(env, env_name),
            )
        except Exception:
            _close_envs(env, eval_env)
            raise

    def prepare_worker_env(seed=None):
        env = env_builder(1, seed=seed)
        return PreparedWorkerEnvSpec(env=env, env_info=_single_env_info(env, env_name))

    env_builder.prepare_envs = prepare_envs
    env_builder.prepare_worker_env = prepare_worker_env
    env_builder.supports_render = True

    env_info = {
        "env_type": "adapter_factory",
        "env_id": env_name,
        "supports_render": env_builder.supports_render,
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
    env_name = env_name.removeprefix("ALE/")

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

    def __init__(self, env_id, worker_num=8, seed=None, observation_key=None):
        import envpool

        self.env_id = env_id
        self.worker_num = worker_num
        self._observation_key = observation_key

        # Convert env_id to EnvPool format
        envpool_env_id = _get_envpool_env_id(env_id)

        # Determine if this is an Atari environment
        self._is_atari = self._check_atari_env(envpool_env_id)
        self._metrics = GymEnvMetrics(worker_num, "reward" if self._is_atari else "original_reward")

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
            "runtime": {
                "backend": "envpool",
                "backend_env_id": envpool_env_id,
                "seed": seed,
                "seed_rule": "backend-managed streams from constructor seed",
                "reward_clipping": "sign" if self._is_atari else "none",
                "episodic_life": self._is_atari,
            },
        }

        # Set up action conversion for the normalized [-1, 1] core contract.
        if not isinstance(self.env.action_space, spaces.Box):
            self.action_conv = lambda a: a.flatten().astype(np.int32)
        elif (
            np.isfinite(self.env.action_space.low).all()
            and np.isfinite(self.env.action_space.high).all()
        ):
            unit = np.ones(self.env.action_space.shape, dtype=self.env.action_space.dtype)
            _, _, self.action_conv = rescale_box(self.env.action_space, -unit, unit)
        else:
            self.action_conv = lambda a: a

        # env_id vector for send(); recv() may hand back envs in completion
        # order, so every result is re-sorted to the canonical 0..N-1 layout
        # the training loop indexes by (scores, prev_done, replay rows).
        self._all_env_ids = np.arange(worker_num, dtype=np.int32)
        self._awaiting_recv = False

        self.reset()

    def _check_atari_env(self, env_id: str) -> bool:
        """Check if the environment is an Atari game."""
        import envpool

        spec = envpool.make_spec(env_id)
        return "Atari" in type(spec).__name__

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        if self._awaiting_recv:
            raise RuntimeError("reset() called while a step is in flight")
        if seed is not None:
            raise ValueError("EnvPool seeds are fixed at construction; recreate the env to reseed")
        self.env.async_reset()
        raw_obs, _, _, _, info = self.env.recv()
        order = np.argsort(info["env_id"])
        self.obs = self._process_observations(raw_obs, order)
        self._metrics.reset()
        return self.obs, self._reorder_info(info, order)

    def step(self, actions):
        """Fire actions into all environments without blocking.

        ``send()`` hands the actions to EnvPool's C++ worker threads and
        returns immediately, so the caller can do useful work (e.g. a gradient
        step) while the environments advance. ``get_result()`` collects the
        outcome via ``recv()``.
        """
        self.env.send(self.action_conv(np.asarray(actions)), self._all_env_ids)
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

        self._metrics.capture(
            infos,
            terminateds | truncateds,
            self.real_reset_mask(terminateds, truncateds, infos),
            self.autoreset_mask(terminateds, truncateds, infos),
        )

        # EnvPool handles auto-reset internally: after a done flag, next_obs
        # already holds the new episode's first observation.
        self.obs = next_obs

        return next_obs, rewards, terminateds, truncateds, infos

    def log_metrics(
        self,
        logger: MetricLogger,
        steps: int | None,
        *,
        namespace: str = "rollout",
        active: Any = None,
        flush: bool = True,
    ) -> None:
        self._metrics.log(logger, steps, namespace=namespace, active=active, flush=flush)

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
        normalized = normalize_observation(obs, self._observation_key)
        if order is not None:
            normalized = {key: value[order] for key, value in normalized.items()}
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
        return normalize_observation_space(obs_space, self._observation_key)


class GymVectorizedEnv(VectorizedEnv):
    """Fallback vectorized environment using gymnasium's native vectorization.

    Used when EnvPool doesn't support the requested environment.
    """

    def __init__(
        self, env_id, worker_num=8, seed=None, observation_key=None, *, reuse_for_eval=False
    ):
        self.env_id = env_id
        self.worker_num = worker_num
        self._observation_key = observation_key
        self._reuse_for_eval = reuse_for_eval
        self._evaluation_active = False
        self._metrics = GymEnvMetrics(worker_num)
        self._pending_result: tuple[Observation, Any, Any, Any, dict[str, Any]] | None = None

        # Create vectorized environment using gymnasium
        # For Atari, we need to use custom wrappers, so we use AsyncVectorEnv with explicit constructors
        from env_builder.atari_wrappers import get_env_type, make_wrap_atari

        env_type, _ = get_env_type(env_id)
        self._is_atari = env_type == "atari_env"
        spec = gym.spec(env_id)

        def make_env():
            def _make():
                if self._is_atari:
                    env = make_wrap_atari(env_id, clip_rewards=True)
                else:
                    env = gym.make(spec)
                env = _normalize_action_space(env)
                if reuse_for_eval:
                    from env_builder.gym_state import GymStateWrapper

                    try:
                        return GymStateWrapper(gym.wrappers.Autoreset(env))
                    except (TypeError, ValueError):
                        env.close()
                        raise
                return env

            return _make

        if reuse_for_eval:
            # Keep NEXT_STEP reset state inside a wrapper that can be snapshotted.
            vector_env = gym.vector.AsyncVectorEnv(
                [make_env() for _ in range(worker_num)],
                context="spawn",
                autoreset_mode=gym.vector.AutoresetMode.DISABLED,
            )
        elif env_type != "atari_env":
            # Non-Atari: prefer the registry's efficient make_vec, falling back to
            # explicit AsyncVectorEnv if the env has no vectorized entry point.
            try:
                vector_env = gym.make_vec(
                    env_id,
                    num_envs=worker_num,
                    vectorization_mode="async",
                    vector_kwargs={"context": "spawn"},
                    wrappers=(_normalize_action_space,),
                )
            except Exception:
                vector_env = gym.vector.AsyncVectorEnv(
                    [make_env() for _ in range(worker_num)], context="spawn"
                )
        else:
            # Atari needs the custom wrappers, so build AsyncVectorEnv from the
            # explicit per-env constructors.
            vector_env = gym.vector.AsyncVectorEnv(
                [make_env() for _ in range(worker_num)], context="spawn"
            )

        if not isinstance(vector_env, gym.vector.AsyncVectorEnv):
            raise TypeError("GymVectorizedEnv requires an AsyncVectorEnv")
        self.env = vector_env

        # Store environment info
        action_size, action_type = _action_meta(self.env.single_action_space)
        self.env_info: EnvInfo = {
            "observation_space": normalize_observation_space(
                self.env.single_observation_space, self._observation_key
            ),
            "action_size": action_size,
            "action_type": action_type,
            "env_type": "gym_vector",
            "env_id": env_id,
            "worker_num": worker_num,
            "core_env_type": "VectorizedEnv",
            "runtime": {
                "backend": "gymnasium",
                "backend_env_id": env_id,
                "seed": seed,
                "seed_rule": "reset seed + worker index",
                "reward_clipping": "sign" if self._is_atari else "none",
                "episodic_life": self._is_atari,
            },
        }

        # Set up action conversion
        if not isinstance(self.env.single_action_space, spaces.Box):
            self.action_conv = lambda a: np.asarray(a).flatten().astype(np.int32)
        else:
            self.action_conv = lambda a: np.asarray(a)

        self._awaiting_result = False
        self.reset(seed=seed)

    def get_info(self):
        return self.env_info

    def current_obs(self):
        return self.obs

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        if self._awaiting_result:
            raise RuntimeError("reset() called while a step is in flight")
        if self._is_atari:
            # Restart the game before outer wrappers rebuild their frame/reward state.
            self.env.set_attr("was_real_done", True)
        obs, info = self.env.reset(seed=seed)
        self.obs = normalize_observation(obs, self._observation_key)
        self._metrics.reset()
        return self.obs, info

    def step(self, actions):
        """Dispatch actions without blocking (gymnasium ``step_async``)."""
        if self._awaiting_result:
            raise RuntimeError("step() called before collecting the preceding result")
        self.env.step_async(self.action_conv(actions))
        self._awaiting_result = True

    def get_result(self):
        """Block on the in-flight async step and return its result."""
        if not self._awaiting_result:
            raise RuntimeError("get_result() called without a preceding step()")
        self._awaiting_result = False
        if self._pending_result is not None:
            result, self._pending_result = self._pending_result, None
            self.obs = result[0]
            return result

        next_obs, rewards, terminateds, truncateds, infos = self.env.step_wait()
        next_obs = normalize_observation(next_obs, self._observation_key)
        self.obs = next_obs
        self._metrics.capture(
            infos,
            terminateds | truncateds,
            self.real_reset_mask(terminateds, truncateds, infos),
            self.autoreset_mask(terminateds, truncateds, infos),
        )

        return next_obs, rewards, terminateds, truncateds, infos

    def log_metrics(
        self,
        logger: MetricLogger,
        steps: int | None,
        *,
        namespace: str = "rollout",
        active: Any = None,
        flush: bool = True,
    ) -> None:
        self._metrics.log(logger, steps, namespace=namespace, active=active, flush=flush)

    @contextmanager
    def evaluation_context(self) -> Iterator[None]:
        if not self._reuse_for_eval:
            metrics = self._metrics
            self._metrics = GymEnvMetrics(self.worker_num)
            try:
                yield
            finally:
                self._metrics = metrics
            return
        if self._evaluation_active:
            raise RuntimeError("Training environment is already borrowed for evaluation")
        from env_builder.gym_state import preserve_gym_spaces

        observation = self.obs
        awaiting = self._awaiting_result
        pending = self.get_result() if awaiting else None
        metrics = self._metrics
        vector_observation = deepcopy(self.env.observations)
        global_rng = random.getstate(), np.random.get_state()
        with preserve_gym_spaces(self.env.action_space, self.env.observation_space):
            self._evaluation_active = True
            try:
                self.env.call("save_training_state")
                try:
                    self._metrics = GymEnvMetrics(self.worker_num)
                    yield
                finally:
                    # Finish an evaluation step even when its action callback raises.
                    if self._awaiting_result:
                        self.get_result()
                    self.env.call("restore_training_state")
            finally:
                concatenate(
                    self.env.single_observation_space,
                    tuple(iterate(self.env.observation_space, vector_observation)),
                    self.env.observations,
                )
                self.obs = observation
                self._pending_result = pending
                self._awaiting_result = awaiting
                self._metrics = metrics
                self._evaluation_active = False
                random.setstate(global_rng[0])
                np.random.set_state(global_rng[1])

    def real_reset_mask(self, terminateds, truncateds, infos):
        return _real_reset_mask(self._is_atari, terminateds, truncateds, infos)

    def autoreset_mask(self, terminateds, truncateds, infos):
        return _autoreset_mask(terminateds, truncateds)

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
