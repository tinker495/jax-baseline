"""Issue #7 audit: distributed workers receive their environment and replay
buffer through injected adapters, never by importing gymnasium / env_builder /
cpprb inside the Algorithm Core.

The static + dynamic import-boundary ratchet (``test_import_boundaries.py``)
proves the forbidden tokens are gone from the core source. These tests prove the
replacement *seam* works: the worker ``__init__`` builds its env from the injected
Environment Adapter callable, and the IMPALA worker-local buffer is produced by
the ``WorkerReplayBufferFactory``. They run without a Ray runtime — the worker
classes are constructed directly (not via ``.remote()``).
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from jax_baselines.APE_X.dpg_worker import Ape_X_Worker as ApexDpgWorker
from jax_baselines.APE_X.worker import Ape_X_Worker as ApexQnetWorker
from jax_baselines.IMPALA.worker import Impala_Worker


class _Space:
    def __init__(self, shape=None, n=None):
        if shape is not None:
            self.shape = shape
        if n is not None:
            self.n = n


class _FakeEnv:
    observation_space = _Space(shape=(4,))
    action_space = _Space(n=2)

    def reset(self, seed=None):
        return np.zeros(4), {}

    def step(self, action):
        return np.zeros(4), 0.0, False, False, {}

    def close(self):
        return None


@pytest.mark.parametrize("worker_cls", [ApexQnetWorker, ApexDpgWorker, Impala_Worker])
def test_worker_builds_env_through_injected_adapter(worker_cls):
    """The worker calls the injected env_builder with ``worker=1`` and seed, and
    exposes the resulting spaces — no in-core gym/env_builder construction."""
    calls = []

    def env_builder(worker=1, seed=None):
        calls.append((worker, seed))
        return _FakeEnv()

    worker = worker_cls(env_builder, seed=7)

    assert calls == [(1, 7)]
    info = worker.get_info()
    assert info["observation_space"].shape == (4,)
    assert info["action_space"].n == 2
    # get_remote_env_info normalizes distributed envs to SingleEnv; the worker
    # reports that directly instead of inferring a backend-specific env_type.
    assert info["env_type"] == "SingleEnv"


def test_impala_worker_replay_factory_satisfies_seam():
    """The IMPALA worker-local buffer is reachable through the core
    ``WorkerReplayBufferFactory`` seam and the experiments composition default."""
    from experiments.replay_factories import make_impala_worker_buffer
    from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer

    env_dict = {
        "obs0": {"shape": (4,)},
        "action": {"shape": 1},
        "log_prob": {},
        "reward": {},
        "next_obs0": {"shape": (4,)},
        "terminated": {},
        "truncted": {},
    }

    buffer = make_worker_local_replay_buffer(make_impala_worker_buffer, 8, env_dict, None)

    assert len(buffer) == 0
    buffer.add(
        [np.zeros((4,))],
        action=0,
        log_prob=0.0,
        reward=1.0,
        nxtobs_t=[np.ones((4,))],
        terminated=False,
    )
    assert len(buffer) == 1
    rollout = buffer.get_buffer()
    # V-trace record carries the behaviour-policy log-prob (mu_log_prob field).
    assert hasattr(rollout, "mu_log_prob")


def test_apex_worker_replay_factory_satisfies_seam():
    """The APE-X worker-local buffer is reachable through the same core
    ``WorkerReplayBufferFactory`` seam and the experiments composition default."""
    from experiments.replay_factories import make_worker_replay_buffer
    from jax_baselines.core.replay_protocol import make_worker_local_replay_buffer

    env_dict = {
        "obs0": {"shape": (4,)},
        "action": {"shape": 1},
        "reward": {},
        "next_obs0": {"shape": (4,)},
        "done": {},
    }

    buffer = make_worker_local_replay_buffer(make_worker_replay_buffer, 8, env_dict, None)

    assert len(buffer) == 0
    buffer.add([np.zeros((4,))], action=0, reward=1.0, nxtobs_t=[np.ones((4,))], terminated=False)
    assert len(buffer) == 1


@pytest.mark.parametrize(
    "relative_path",
    [
        "jax_baselines/IMPALA/base_class.py",
        "jax_baselines/APE_X/base_class.py",
        "jax_baselines/APE_X/dpg_base_class.py",
    ],
)
def test_distributed_base_class_threads_worker_replay_factory(relative_path):
    """Regression: every distributed learner must thread ``worker_replay_factory``
    into ``worker.run.remote(...)`` so the worker never falls back to a core
    buffer. IMPALA passes a local variable; APE-X passes
    ``self.worker_replay_factory`` — accept a Name or an attribute access."""
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / relative_path).read_text())

    run_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "run"
    )
    passed = {
        arg.id if isinstance(arg, ast.Name) else arg.attr
        for arg in run_call.args
        if isinstance(arg, (ast.Name, ast.Attribute))
    }
    assert (
        "worker_replay_factory" in passed
    ), f"{relative_path}: run.remote must pass worker_replay_factory"
