import warnings
from collections.abc import Mapping

import cpprb
import numpy as np


def _obs_spec(shape):
    """uint8 spec for image-like (>=3 dims) features, float32 otherwise."""
    return (
        {"shape": shape, "dtype": np.uint8}
        if len(shape) >= 3
        else {"shape": shape, "dtype": np.float32}
    )


def _build_obs_dicts(observation_space):
    """Build cpprb fields while preserving canonical observation keys."""
    if not isinstance(observation_space, Mapping):
        raise TypeError("observation_space must be a dict")
    obsdict = {f"obs:{key}": _obs_spec(shape) for key, shape in observation_space.items()}
    nextobsdict = {f"next_obs:{key}": _obs_spec(shape) for key, shape in observation_space.items()}
    return obsdict, nextobsdict


def _storage_observation(fields, observation, prefix):
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a dict")
    return {field: observation[field.removeprefix(f"{prefix}:")] for field in fields}


def _compress_config(obsdict, nextobsdict, compress_memory, n_step):
    """Resolve cpprb memory-compression settings for image observations.

    Returns ``(env_nextobsdict, cpprb_kwargs)`` where ``env_nextobsdict`` is the
    next-observation portion to declare in the cpprb ``env_dict`` and
    ``cpprb_kwargs`` carries ``next_of`` / ``stack_compress``. The wrapper always
    keeps the full ``next_obs`` keys for ``add`` / ``sample`` iteration; only the
    env_dict declaration is pruned, because ``next_of`` makes cpprb recreate the
    compressed ``next_obs`` automatically.

    Verified cpprb behaviour (cpprb 10.x, see tests/test_cpprb_compress.py):
      * ``next_of`` shares next-observation memory with the observation but is
        incompatible with ``Nstep`` (the moved ``next_*`` field disappears from
        ``sample``), so it is only used on the single-step path.
      * ``stack_compress`` drops the duplicated frames of a frame-stacked
        observation and reconstructs them from sequentially stored rows. It must
        not cover the n-step ``next_obs`` (its rows are n steps apart, breaking
        reconstruction); on the n-step path ``next_obs`` stays fully stored so
        ``Nstep`` can move it.
    """
    image_keys = [k for k in obsdict if len(obsdict[k]["shape"]) >= 3]
    if not compress_memory or not image_keys:
        return dict(nextobsdict), {}
    if n_step > 1:
        return dict(nextobsdict), {"stack_compress": image_keys}
    env_nextobsdict = {
        k: v for k, v in nextobsdict.items() if k.removeprefix("next_") not in image_keys
    }
    return env_nextobsdict, {"next_of": image_keys, "stack_compress": image_keys}


def _nstep_env_dicts(obsdict, nextobsdict, env_nextobsdict, action_space, n_step, gamma):
    """Build the cpprb ``Nstep`` config and the central / local ``env_dict`` pair
    shared by the (Prioritized)NstepReplayBuffer constructors."""
    n_s = {
        "size": n_step,
        "rew": "reward",
        "gamma": gamma,
        "next": list(nextobsdict.keys()),
    }
    central_env_dict = {
        **obsdict,
        "action": {"shape": action_space},
        "reward": {},
        **env_nextobsdict,
        "done": {},
    }
    local_env_dict = {
        **obsdict,
        "action": {"shape": action_space},
        "reward": {},
        **nextobsdict,
        "done": {},
    }
    return n_s, central_env_dict, local_env_dict


def _active_worker_indices(worker_size, store_mask):
    if store_mask is None:
        return range(worker_size)

    store_mask = np.asarray(store_mask, dtype=bool)
    if store_mask.shape != (worker_size,):
        raise ValueError("store_mask length must match worker_size")
    if store_mask.all():
        return range(worker_size)
    return np.flatnonzero(store_mask)


def _terminated_mask(done):
    return np.clip(done, 0.0, 1.0)


def _project_transitions(transitions, obs_keys, next_obs_keys, *, prioritized=False):
    projected = {
        "obses": {key.removeprefix("obs:"): transitions[key] for key in obs_keys},
        "actions": transitions["action"],
        "rewards": transitions["reward"],
        "nxtobses": {key.removeprefix("next_obs:"): transitions[key] for key in next_obs_keys},
        "terminateds": _terminated_mask(transitions["done"]),
    }
    if prioritized:
        projected["weights"] = transitions["weights"]
        projected["indexes"] = transitions["indexes"]
    return projected


class ReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = None,
        action_space=1,
        compress_memory=False,
        env_dict=None,
        n_s=None,
    ):
        self.max_size = size
        if env_dict is None:
            self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space or [])
            env_nextobsdict, comp_kw = _compress_config(
                self.obsdict, self.nextobsdict, compress_memory, n_step=1
            )
            self.buffer = cpprb.ReplayBuffer(
                size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **env_nextobsdict,
                    "done": {},
                },
                **comp_kw,
            )
            self._compress_active = bool(comp_kw)
        else:
            self.obsdict = {o: None for o in env_dict if o.startswith("obs")}
            self.nextobsdict = {o: None for o in env_dict if o.startswith("next_obs")}
            self.buffer = cpprb.ReplayBuffer(size, env_dict=env_dict, Nstep=n_s)
            self._compress_active = False

    def __len__(self) -> int:
        return self.buffer.get_stored_size()

    def add(
        self,
        obs_t,
        action,
        reward,
        nxtobs_t,
        terminated,
        truncated=False,
        store_mask=None,
    ):
        # store_mask drops vectorized workers whose previous step ended an
        # episode: their current step is an autoreset dummy (action ignored,
        # reward 0, fresh obs) that must not enter the buffer.
        if store_mask is not None:
            if not store_mask.any():
                return
            obs_t = {key: value[store_mask] for key, value in obs_t.items()}
            nxtobs_t = {key: value[store_mask] for key, value in nxtobs_t.items()}
            action = action[store_mask]
            reward = reward[store_mask]
            terminated = terminated[store_mask]
            truncated = truncated[store_mask]
        obsdict = _storage_observation(self.obsdict, obs_t, "obs")
        nextobsdict = _storage_observation(self.nextobsdict, nxtobs_t, "next_obs")
        self.buffer.add(**obsdict, action=action, reward=reward, **nextobsdict, done=terminated)
        # next_of / stack_compress reconstruct observations from sequentially
        # stored rows, so the episode boundary must be marked or the terminal
        # frame window would bleed into the next episode. Compression assumes a
        # single contiguous stream here; multi-worker image compression is routed
        # through NstepReplayBuffer.multiworker_single_step_add instead.
        if self._compress_active and (terminated or truncated):
            self.buffer.on_episode_end()

    def episode_end(self):
        self.buffer.on_episode_end()

    def sample(self, batch_size: int):
        return _project_transitions(self.buffer.sample(batch_size), self.obsdict, self.nextobsdict)

    def get_buffer(self):
        return self.buffer.get_all_transitions()

    def conv_transitions(self, transitions):
        return _project_transitions(transitions, self.obsdict, self.nextobsdict)

    def clear(self):
        self.buffer.clear()


class NstepReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        action_space=1,
        worker_size=1,
        n_step=1,
        gamma=0.99,
        compress_memory=False,
    ):
        self.max_size = size
        self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
        env_nextobsdict, comp_kw = _compress_config(
            self.obsdict, self.nextobsdict, compress_memory, n_step=n_step
        )
        # The n-step add() marks episode boundaries itself; keep the base add()
        # from doing it a second time.
        self._compress_active = False
        self.worker_size = worker_size
        n_s, central_env_dict, local_env_dict = _nstep_env_dicts(
            self.obsdict, self.nextobsdict, env_nextobsdict, action_space, n_step, gamma
        )

        if worker_size > 1:
            self.buffer = self._create_central_buffer(size, central_env_dict, comp_kw)
            # ponytail: row isolation preserves immediate sampling;
            # per-worker central buffers if compression ratio matters.
            self._isolate_multiworker_steps = bool(comp_kw)
            if n_step == 1:
                self.add = self.multiworker_single_step_add
            else:
                self._local_capacity = n_step + 1
                self.local_buffers = [
                    cpprb.ReplayBuffer(self._local_capacity, env_dict=local_env_dict, Nstep=n_s)
                    for _ in range(worker_size)
                ]
                self.add = self.multiworker_add
        else:
            self.buffer = self._create_central_buffer(size, central_env_dict, comp_kw, n_s=n_s)

    def _create_central_buffer(self, size, env_dict, comp_kw, n_s=None):
        if n_s is not None:
            comp_kw = {**comp_kw, "Nstep": n_s}
        return cpprb.ReplayBuffer(size, env_dict=env_dict, **comp_kw)

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        super().add(obs_t, action, reward, nxtobs_t, terminated, truncated)
        if terminated or truncated:
            self.buffer.on_episode_end()

    def multiworker_add(
        self,
        obs_t,
        action,
        reward,
        nxtobs_t,
        terminated,
        truncated=False,
        store_mask=None,
    ):
        for w in _active_worker_indices(self.worker_size, store_mask):
            obsdict = _storage_observation(
                self.obsdict, {key: value[w] for key, value in obs_t.items()}, "obs"
            )
            nextobsdict = _storage_observation(
                self.nextobsdict,
                {key: value[w] for key, value in nxtobs_t.items()},
                "next_obs",
            )
            local_buffer = self.local_buffers[w]
            start = local_buffer.get_next_index()
            local_buffer.add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                done=terminated[w],
            )
            boundary = terminated[w] or truncated[w]
            if boundary:
                local_buffer.on_episode_end()

            count = (local_buffer.get_next_index() - start) % self._local_capacity
            if count:
                indexes = (start + np.arange(count)) % self._local_capacity
                transitions = {
                    key: value[indexes] for key, value in local_buffer.get_all_transitions().items()
                }
                transitions["done"] = _terminated_mask(transitions["done"])
                self.buffer.add(**transitions)

            if count and (self._isolate_multiworker_steps or boundary):
                self.buffer.on_episode_end()
            if boundary:
                local_buffer.clear()

    def multiworker_single_step_add(
        self,
        obs_t,
        action,
        reward,
        nxtobs_t,
        terminated,
        truncated=False,
        store_mask=None,
    ):
        for w in _active_worker_indices(self.worker_size, store_mask):
            obsdict = _storage_observation(
                self.obsdict, {key: value[w] for key, value in obs_t.items()}, "obs"
            )
            nextobsdict = _storage_observation(
                self.nextobsdict,
                {key: value[w] for key, value in nxtobs_t.items()},
                "next_obs",
            )
            self.buffer.add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                done=terminated[w],
            )
            if self._isolate_multiworker_steps or terminated[w] or truncated[w]:
                self.buffer.on_episode_end()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        alpha: float,
        action_space=1,
        compress_memory=False,
        eps=1e-4,
    ):
        self.max_size = size
        self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
        env_nextobsdict, comp_kw = _compress_config(
            self.obsdict, self.nextobsdict, compress_memory, n_step=1
        )
        self._compress_active = bool(comp_kw)
        self.buffer = cpprb.PrioritizedReplayBuffer(
            size,
            env_dict={
                **self.obsdict,
                "action": {"shape": action_space},
                "reward": {},
                **env_nextobsdict,
                "done": {},
            },
            alpha=alpha,
            eps=eps,
            **comp_kw,
        )

    def sample(self, batch_size: int, beta=0.5):
        return _project_transitions(
            self.buffer.sample(batch_size, beta),
            self.obsdict,
            self.nextobsdict,
            prioritized=True,
        )

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)


class PrioritizedNstepReplayBuffer(NstepReplayBuffer):
    def __init__(
        self,
        size: int,
        observation_space: list,
        action_space=1,
        worker_size=1,
        n_step=1,
        gamma=0.99,
        alpha=0.4,
        compress_memory=False,
        eps=1e-4,
    ):
        self.alpha = alpha
        self.eps = eps
        super().__init__(
            size=size,
            observation_space=observation_space,
            action_space=action_space,
            worker_size=worker_size,
            n_step=n_step,
            gamma=gamma,
            compress_memory=compress_memory,
        )

    def _create_central_buffer(self, size, env_dict, comp_kw, n_s=None):
        if n_s is not None:
            comp_kw = {**comp_kw, "Nstep": n_s}
        return cpprb.PrioritizedReplayBuffer(
            size, env_dict=env_dict, alpha=self.alpha, eps=self.eps, **comp_kw
        )

    def sample(self, batch_size: int, beta=0.5):
        return _project_transitions(
            self.buffer.sample(batch_size, beta),
            self.obsdict,
            self.nextobsdict,
            prioritized=True,
        )

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)


class MultiPrioritizedReplayBuffer:
    def __init__(
        self,
        size: int,
        observation_space: list,
        alpha: float,
        action_space=1,
        n_step=1,
        gamma=0.99,
        manager=None,
        compress_memory=False,
        eps=1e-4,
    ):
        self.max_size = size
        self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
        if compress_memory and any(len(spec["shape"]) >= 3 for spec in self.obsdict.values()):
            warnings.warn(
                "compress_memory is unsupported for the distributed APE-X buffer: "
                "the shared central buffer is fed shuffled, batched n-step "
                "transitions, which violates the sequential sliding-window "
                "assumption of cpprb stack/next compression. Image observations "
                "are still stored as uint8; the flag is ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.env_dict = {
            **self.obsdict,
            "action": {"shape": action_space},
            "reward": {},
            **self.nextobsdict,
            "done": {},
        }

        self.n_s = None
        if n_step > 1:
            self.n_s = {
                "size": n_step,
                "rew": "reward",
                "gamma": gamma,
                "next": list(self.nextobsdict.keys()),
            }

        self.buffer = cpprb.MPPrioritizedReplayBuffer(
            size,
            env_dict=self.env_dict,
            alpha=alpha,
            eps=eps,
            ctx=manager,
            backend="SharedMemory",
        )

    def __len__(self):
        return self.buffer.get_stored_size()

    def buffer_info(self):
        return self.buffer, self.env_dict, self.n_s

    def sample(self, batch_size: int, beta=0.5):
        return _project_transitions(
            self.buffer.sample(batch_size, beta),
            self.obsdict,
            self.nextobsdict,
            prioritized=True,
        )

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)
