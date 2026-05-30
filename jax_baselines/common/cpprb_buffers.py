import warnings

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
    """Build the ``obs{i}`` / ``next_obs{i}`` cpprb env_dict fragments."""
    obsdict = {f"obs{idx}": _obs_spec(o) for idx, o in enumerate(observation_space)}
    nextobsdict = {f"next_obs{idx}": _obs_spec(o) for idx, o in enumerate(observation_space)}
    return obsdict, nextobsdict


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


class EpochBuffer(object):
    def __init__(self, epoch_size: int, observation_space: list, worker_size=1, action_space=1):
        self.epoch_size = epoch_size
        self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
        self.worker_size = worker_size
        self.local_buffers = [
            cpprb.ReplayBuffer(
                epoch_size,
                env_dict={
                    **self.obsdict,
                    "action": {"shape": action_space},
                    "reward": {},
                    **self.nextobsdict,
                    "terminated": {},
                    "truncated": {},
                },
            )
            for _ in range(worker_size)
        ]

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated):
        for w in range(self.worker_size):
            obsdict = dict(zip(self.obsdict.keys(), [o[w] for o in obs_t]))
            nextobsdict = dict(zip(self.nextobsdict.keys(), [no[w] for no in nxtobs_t]))
            self.local_buffers[w].add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                terminated=terminated[w],
                truncated=truncated[w],
            )
            if terminated[w] or truncated[w]:
                self.local_buffers[w].on_episode_end()

    def get_buffer(self):
        transitions = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "nxtobses": [],
            "terminateds": [],
            "truncateds": [],
        }
        for w in range(self.worker_size):
            trans = self.local_buffers[w].get_all_transitions()
            transitions["obses"].append([trans[o] for o in self.obsdict.keys()])
            transitions["actions"].append(trans["action"])
            transitions["rewards"].append(trans["reward"])
            transitions["nxtobses"].append([trans[o] for o in self.nextobsdict.keys()])
            transitions["terminateds"].append(trans["terminated"])
            transitions["truncateds"].append(trans["truncated"])
            self.local_buffers[w].clear()
        return transitions


class ReplayBuffer(object):
    def __init__(
        self,
        size: int,
        observation_space: list = [],
        action_space=1,
        compress_memory=False,
        env_dict=None,
        n_s=None,
    ):
        self.max_size = size
        if env_dict is None:
            self.obsdict, self.nextobsdict = _build_obs_dicts(observation_space)
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
            self.obsdict = dict((o, None) for o in env_dict.keys() if o.startswith("obs"))
            self.nextobsdict = dict((o, None) for o in env_dict.keys() if o.startswith("next_obs"))
            self.buffer = cpprb.ReplayBuffer(size, env_dict=env_dict, Nstep=n_s)
            self._compress_active = False

    def __len__(self) -> int:
        return self.buffer.get_stored_size()

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        obsdict = dict(zip(self.obsdict.keys(), obs_t))
        nextobsdict = dict(zip(self.nextobsdict.keys(), nxtobs_t))
        self.buffer.add(**obsdict, action=action, reward=reward, **nextobsdict, done=terminated)
        # next_of / stack_compress reconstruct observations from sequentially
        # stored rows, so the episode boundary must be marked or the terminal
        # frame window would bleed into the next episode.
        if self._compress_active and (terminated or truncated):
            self.buffer.on_episode_end()

    def episode_end(self):
        self.buffer.on_episode_end()

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["done"],
        }

    def get_buffer(self):
        return self.buffer.get_all_transitions()

    def conv_transitions(self, transitions):
        return {
            "obses": [transitions[o] for o in self.obsdict.keys()],
            "actions": transitions["action"],
            "rewards": transitions["reward"],
            "nxtobses": [transitions[no] for no in self.nextobsdict.keys()],
            "terminateds": transitions["done"],
        }

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
            self.buffer = cpprb.ReplayBuffer(size, env_dict=central_env_dict, **comp_kw)
            self.local_buffers = [
                cpprb.ReplayBuffer(2000, env_dict=local_env_dict, Nstep=n_s)
                for _ in range(worker_size)
            ]
            self.add = self.multiworker_add
        else:
            self.buffer = cpprb.ReplayBuffer(size, env_dict=central_env_dict, Nstep=n_s, **comp_kw)

    def add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        super().add(obs_t, action, reward, nxtobs_t, terminated, truncated)
        if terminated or truncated:
            self.buffer.on_episode_end()

    def multiworker_add(self, obs_t, action, reward, nxtobs_t, terminated, truncated=False):
        for w in range(self.worker_size):
            obsdict = dict(zip(self.obsdict.keys(), [o[w] for o in obs_t]))
            nextobsdict = dict(zip(self.nextobsdict.keys(), [no[w] for no in nxtobs_t]))
            self.local_buffers[w].add(
                **obsdict,
                action=action[w],
                reward=reward[w],
                **nextobsdict,
                done=terminated[w],
            )
            if terminated[w] or truncated[w]:
                self.local_buffers[w].on_episode_end()
                self.buffer.add(**self.local_buffers[w].get_all_transitions())
                # Mark the episode boundary so stack_compress reconstructs each
                # worker's frame window without bleeding across episodes.
                self.buffer.on_episode_end()
                self.local_buffers[w].clear()


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
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["done"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

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
            self.buffer = cpprb.PrioritizedReplayBuffer(
                size, env_dict=central_env_dict, alpha=alpha, eps=eps, **comp_kw
            )
            self.local_buffers = [
                cpprb.ReplayBuffer(2000, env_dict=local_env_dict, Nstep=n_s)
                for _ in range(worker_size)
            ]
            self.add = self.multiworker_add
        else:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                size, env_dict=central_env_dict, alpha=alpha, eps=eps, Nstep=n_s, **comp_kw
            )

    def sample(self, batch_size: int, beta=0.5):
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["done"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

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
        smpl = self.buffer.sample(batch_size, beta)
        return {
            "obses": [smpl[o] for o in self.obsdict.keys()],
            "actions": smpl["action"],
            "rewards": smpl["reward"],
            "nxtobses": [smpl[no] for no in self.nextobsdict.keys()],
            "terminateds": smpl["done"],
            "weights": smpl["weights"],
            "indexes": smpl["indexes"],
        }

    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes, priorities)
