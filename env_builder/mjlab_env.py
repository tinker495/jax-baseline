"""Lazy mjlab adapters preserving terminal observations across partial resets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from env_builder.observations import _to_numpy, normalize_observation
from jax_baselines.core.env_protocols import (
    EnvInfo,
    Observation,
    ObservationSpace,
    SingleEnv,
    VectorizedEnv,
)

Array: TypeAlias = np.ndarray | jax.Array


class MjlabVectorizedEnv(VectorizedEnv):
    env_info: EnvInfo

    def __init__(self, env_id, env, torch, seed=None, observation_key=None, jax_arrays=False):
        self.env = env
        self._torch = torch
        self.jax_arrays = jax_arrays
        self.worker_num = env.num_envs
        self._observation_key = observation_key
        self._pending: tuple[Observation, Array, Array, Array, dict[str, Any]] | None = None
        self._closed = False
        self._frame = None
        self.reset(seed=seed)
        shape = tuple(env.single_action_space.shape)
        self.action_space = spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)
        self.observation_space: ObservationSpace = {
            key: list(value.shape[1:]) for key, value in self._obs.items()
        }
        self.env_info = {
            "observation_space": self.observation_space,
            "action_size": list(shape),
            "action_type": "continuous",
            "env_type": "mjlab",
            "env_id": env_id,
            "worker_num": self.worker_num,
            "core_env_type": "VectorizedEnv",
        }

    def _array(self, value) -> Array:
        if not self.jax_arrays:
            return _to_numpy(value).copy()
        if isinstance(value, self._torch.Tensor):
            # Clone on the producer stream before mjlab mutates/reset its buffers.
            return jax.dlpack.from_dlpack(
                value.detach().clone(memory_format=self._torch.contiguous_format)
            )
        return jnp.array(value, copy=True)

    def _snapshot(self, value):
        if isinstance(value, Mapping):
            return {key: self._snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._snapshot(item) for item in value)
        if isinstance(value, (self._torch.Tensor, np.ndarray, jax.Array)):
            return self._array(value)
        return value

    def _selected(self, observation) -> Observation:
        if (
            self._observation_key is None
            and isinstance(observation, Mapping)
            and {"actor", "critic"} <= observation.keys()
        ):
            shared = {
                key: value for key, value in observation.items() if key not in ("actor", "critic")
            }
            selected = normalize_observation(shared, array_converter=self._array) if shared else {}
            for role in ("actor", "critic"):
                selected.update(
                    {
                        f"{role}_{key.removeprefix('unified_')}": value
                        for key, value in normalize_observation(
                            observation[role], array_converter=self._array
                        ).items()
                    }
                )
            return dict(sorted(selected.items()))
        return normalize_observation(
            observation, self._observation_key, array_converter=self._array
        )

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        if self._pending is not None:
            raise RuntimeError("reset() called while a step is in flight")
        self._frame = None
        observation, info = self.env.reset(seed=seed)
        self._obs = self._selected(observation)
        return self._obs, self._snapshot(info)

    def current_obs(self) -> Observation:
        return self._obs

    def get_info(self) -> EnvInfo:
        return self.env_info

    def step(self, action: Any) -> None:
        if self._pending is not None:
            raise RuntimeError("step() called before collecting the previous result")
        actions = (
            self._torch.from_dlpack(action)
            if isinstance(action, jax.Array)
            else self._torch.as_tensor(action)
        )
        expected = (self.worker_num, *self.action_space.shape)
        if actions.shape != expected:
            raise ValueError(f"Expected actions with shape {expected}, got {actions.shape}")
        actions = actions.to(dtype=self._torch.float32, device=self.env.device)
        observation, reward, terminated, truncated, info = self.env.step(actions)
        if self.env.render_mode == "rgb_array":
            self._frame = np.array(self.env.render(), copy=True)
        # Snapshot terminal data before partial reset mutates simulator buffers.
        successor = self._selected(observation)
        done_ids = self._torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        reward = self._array(reward)
        terminated = self._array(terminated).astype(bool)
        truncated = self._array(truncated).astype(bool)
        info = self._snapshot(info)
        self._obs = successor
        if done_ids.numel():
            current, _ = self.env.reset(env_ids=done_ids)
            # Only replace reset rows: a partial reset may recompute noisy observations.
            current = self._selected(current)
            array_module = jnp if self.jax_arrays else np
            done = terminated | truncated
            self._obs = {
                key: array_module.where(
                    done.reshape((self.worker_num,) + (1,) * (value.ndim - 1)),
                    current[key],
                    value,
                )
                for key, value in successor.items()
            }
        self._pending = successor, reward, terminated, truncated, info

    def get_result(self) -> tuple[Observation, Array, Array, Array, dict[str, Any]]:
        if self._pending is None:
            raise RuntimeError("get_result() called without a preceding step()")
        result, self._pending = self._pending, None
        return result

    def real_reset_mask(self, terminateds, truncateds, infos):
        del infos
        return terminateds | truncateds

    def autoreset_mask(self, terminateds, truncateds, infos):
        del truncateds, infos
        return (jnp if self.jax_arrays else np).zeros_like(terminateds, dtype=bool)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.env.close()


class MjlabSingleEnv(SingleEnv):
    def __init__(self, vector: MjlabVectorizedEnv):
        self._vector = vector
        self.render_mode = vector.env.render_mode
        self.metadata = dict(vector.env.metadata)
        self.observation_space = vector.observation_space
        self.action_space = vector.action_space
        self._cached_reset: Observation | None = self._current()

    def _current(self) -> Observation:
        return {key: value[0].copy() for key, value in self._vector.current_obs().items()}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        del options
        self._vector._frame = None
        if seed is None and self._cached_reset is not None:
            observation, self._cached_reset = self._cached_reset, None
            return observation, {}
        self._cached_reset = None
        _, info = self._vector.reset(seed=seed)
        return self._current(), info

    def step(self, action: Any) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        action = (jnp if self._vector.jax_arrays else np).asarray(action)
        self._vector.step(action[None])
        observation, reward, terminated, truncated, info = self._vector.get_result()
        if terminated[0] or truncated[0]:
            self._cached_reset = self._current()
        return (
            {key: value[0] for key, value in observation.items()},
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )

    def render(self):
        if self._vector._frame is not None:
            return self._vector._frame
        return self._vector.env.render()

    def close(self) -> None:
        self._vector.close()


def make_mjlab_env(
    env_id,
    worker_num=1,
    seed=None,
    observation_key=None,
    episode_length=None,
    device="cuda:0",
    render_mode=None,
    jax_arrays=False,
) -> MjlabSingleEnv | MjlabVectorizedEnv:
    if render_mode not in (None, "rgb_array"):
        raise ValueError("mjlab supports only render_mode=None or 'rgb_array'")
    if worker_num < 1:
        raise ValueError("worker_num must be at least 1")
    if episode_length is not None and episode_length < 1:
        raise ValueError("episode_length must be at least 1")
    try:
        import mjlab.tasks  # noqa: F401 - registers built-in tasks
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg
    except ImportError as exc:
        raise ImportError(
            "mjlab is required; install the 'mjlab' extra with Python 3.11–3.13"
        ) from exc

    cfg = load_env_cfg(env_id)
    cfg.scene.num_envs = worker_num
    cfg.seed = seed
    cfg.auto_reset = False
    if episode_length is not None:
        cfg.episode_length_s = episode_length * (cfg.sim.mujoco.timestep * cfg.decimation)
    env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)
    try:
        vector = MjlabVectorizedEnv(env_id, env, torch, seed, observation_key, jax_arrays)
        return MjlabSingleEnv(vector) if worker_num == 1 else vector
    except Exception:
        env.close()
        raise
