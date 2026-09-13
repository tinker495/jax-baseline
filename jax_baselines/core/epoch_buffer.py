"""Worker-major on-policy batches stored in CPU or GPU memory."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

_Array = np.ndarray | jax.Array
OldPolicy = tuple[_Array, _Array | tuple[_Array, _Array]]


class EpochBatch(TypedDict):
    obses: dict[str, np.ndarray | jax.Array]
    actions: np.ndarray | jax.Array
    rewards: np.ndarray | jax.Array
    nxtobses: dict[str, np.ndarray | jax.Array]
    terminateds: np.ndarray | jax.Array
    truncateds: np.ndarray | jax.Array
    old_policy: NotRequired[OldPolicy]


@jax.jit
def _stack_transitions(transitions: list[EpochBatch]) -> EpochBatch:
    return jax.tree.map(lambda *values: jnp.stack(values, axis=1), *transitions)


class EpochBuffer:
    def __init__(
        self,
        epoch_size: int,
        observation_space: dict,
        worker_size=1,
        action_space=1,
        memory_backend: Literal["cpu", "gpu"] = "cpu",
        memory_device: jax.Device | None = None,
    ):
        if epoch_size < 1 or worker_size < 1:
            raise ValueError("epoch_size and worker_size must be positive")
        if not isinstance(observation_space, dict) or not observation_space:
            raise ValueError("observation_space must be a non-empty dict")
        if memory_backend not in ("cpu", "gpu"):
            raise ValueError("memory_backend must be 'cpu' or 'gpu'")
        if memory_device is not None and memory_device.platform != memory_backend:
            raise ValueError("memory_device must match memory_backend")
        self.memory_backend = memory_backend
        self._device = memory_device
        if memory_backend == "gpu" and self._device is None:
            try:
                devices = jax.devices("gpu")
            except RuntimeError as error:
                raise RuntimeError(
                    "GPU epoch memory requires an available JAX GPU device"
                ) from error
            if not devices:
                raise RuntimeError("GPU epoch memory requires an available JAX GPU device")
            self._device = devices[0]
        self.epoch_size = epoch_size
        self.observation_space = observation_space
        self.worker_size = worker_size
        self.action_shape = (
            (action_space,) if isinstance(action_space, int) else tuple(action_space)
        )
        self._transitions: list[EpochBatch] = []

    def add(
        self,
        obs_t,
        action,
        reward,
        nxtobs_t,
        terminated,
        truncated,
        *,
        old_policy: OldPolicy | None = None,
    ):
        if len(self._transitions) >= self.epoch_size:
            raise ValueError("EpochBuffer is full; consume the rollout before adding transitions")
        if (
            not isinstance(obs_t, dict)
            or not isinstance(nxtobs_t, dict)
            or obs_t.keys() != self.observation_space.keys()
            or nxtobs_t.keys() != obs_t.keys()
        ):
            raise ValueError("Observation keys must match observation_space")
        transition: EpochBatch
        if self.memory_backend == "cpu":
            transition = {
                "obses": {key: np.array(value, copy=True) for key, value in obs_t.items()},
                "actions": np.array(action, copy=True),
                "rewards": np.array(reward, copy=True),
                "nxtobses": {key: np.array(value, copy=True) for key, value in nxtobs_t.items()},
                "terminateds": np.array(terminated, dtype=bool, copy=True),
                "truncateds": np.array(truncated, dtype=bool, copy=True),
            }
        else:
            transition = jax.tree.map(
                lambda value: jnp.array(
                    value, copy=not isinstance(value, jax.Array), device=self._device
                ),
                {
                    "obses": obs_t,
                    "actions": action,
                    "rewards": reward,
                    "nxtobses": nxtobs_t,
                    "terminateds": terminated,
                    "truncateds": truncated,
                },
                is_leaf=lambda value: isinstance(value, (list, tuple)),
            )
        if old_policy is not None:
            log_prob, distribution = old_policy
            if log_prob.shape != (self.worker_size, 1):
                raise ValueError(
                    f"Expected old-policy log probabilities with shape {(self.worker_size, 1)}"
                )
            if isinstance(distribution, tuple):
                if len(distribution) != 2 or any(
                    value.shape != (self.worker_size, *self.action_shape) for value in distribution
                ):
                    raise ValueError("Expected old-policy mean and log std with the action shape")
            elif (
                distribution.ndim != 2
                or distribution.shape[0] != self.worker_size
                or distribution.shape[1] < 1
            ):
                raise ValueError("Expected old-policy probabilities with shape (workers, actions)")
            transition["old_policy"] = (
                jax.tree.map(lambda value: np.array(value, copy=True), old_policy)
                if self.memory_backend == "cpu"
                else jax.tree.map(
                    lambda value: jnp.array(
                        value,
                        copy=not isinstance(value, jax.Array),
                        device=self._device,
                    ),
                    old_policy,
                )
            )
        transition["rewards"] = transition["rewards"].reshape(self.worker_size)
        transition["terminateds"] = (
            transition["terminateds"].astype(bool, copy=False).reshape(self.worker_size)
        )
        transition["truncateds"] = (
            transition["truncateds"].astype(bool, copy=False).reshape(self.worker_size)
        )
        for key, shape in self.observation_space.items():
            expected = (self.worker_size, *shape)
            if (
                transition["obses"][key].shape != expected
                or transition["nxtobses"][key].shape != expected
            ):
                raise ValueError(f"Expected observation {key!r} with shape {expected}")
        if transition["actions"].shape != (self.worker_size, *self.action_shape):
            raise ValueError(
                f"Expected actions with shape {(self.worker_size, *self.action_shape)}"
            )
        if self._transitions and (
            jax.tree.structure(transition) != jax.tree.structure(self._transitions[0])
            or [value.shape for value in jax.tree.leaves(transition)]
            != [value.shape for value in jax.tree.leaves(self._transitions[0])]
        ):
            raise ValueError(
                "Transition structure and shapes must remain constant within a rollout"
            )
        self._transitions.append(transition)

    def get_buffer(self) -> EpochBatch:
        if not self._transitions:
            raise ValueError("Cannot consume an empty EpochBuffer")
        if self.memory_backend == "cpu":
            transitions: EpochBatch = jax.tree.map(
                lambda *values: np.stack(values, axis=1), *self._transitions
            )
        else:
            transitions = _stack_transitions(self._transitions)
        self._transitions.clear()
        return transitions
