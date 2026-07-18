"""Regression tests for the EnvPool / gymnasium vectorized-env fixes.

Three independent fixes are pinned here:

- Fix 1 — CLI default env ids must be gymnasium-canonical. The eval env always
  uses the single-env ``gym.make`` path, so an envpool-only id (``Pendulum-v0``)
  or a typo (``Cartpole-v1``) crashes a ``--worker N>1`` run before training.
- Fix 2 — vectorized rollout must drop the autoreset *dummy* step. EnvPool and
  gymnasium vector envs reset on the step *after* an episode ends (action
  ignored, reward 0, fresh obs); that bogus terminal->reset transition must not
  enter the replay buffer. The buffers honor a ``store_mask``.
- Fix 3 — the vectorized backend is an explicit ``--env_backend`` choice
  (default gymnasium; envpool opt-in). Requesting envpool for an env it cannot
  build fails fast instead of silently degrading.
"""

import argparse
import importlib
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from env_builder import env_builder as eb
from replay_memory.cpprb_buffers import (
    NstepReplayBuffer,
    ReplayBuffer,
    _active_worker_indices,
)


# --- Fix 1: CLI default env ids are gymnasium-canonical ------------------
@pytest.mark.parametrize("mod_name", ["qnet", "dpg", "apex_qnet", "apex_dpg"])
def test_cli_default_env_is_gym_registered(mod_name):
    mod = importlib.import_module(f"experiments.cli.{mod_name}")
    parser = argparse.ArgumentParser()
    mod.add_args(parser)
    default_env = parser.parse_known_args([])[0].env
    # Raises DeprecatedEnv / NameNotFound for envpool-only ids or typos, which
    # is exactly the eval-env crash these defaults used to trigger.
    gym.spec(default_env)


# --- Fix 2: ReplayBuffer store_mask drops post-done dummy workers ---------
def _batch(worker_size):
    obs = np.arange(worker_size * 2, dtype=np.float32).reshape(worker_size, 2)
    nxt = obs + 0.5
    act = np.arange(worker_size, dtype=np.float32).reshape(worker_size, 1)
    rew = np.ones(worker_size, dtype=np.float32)
    term = np.zeros(worker_size, dtype=bool)
    trunc = np.zeros(worker_size, dtype=bool)
    return obs, act, rew, nxt, term, trunc


def test_replay_buffer_no_mask_adds_every_worker():
    buf = ReplayBuffer(100, {"obs": [2]}, 1)
    obs, act, rew, nxt, term, trunc = _batch(3)
    buf.add({"obs": obs}, act, rew, {"obs": nxt}, term, trunc)
    assert len(buf) == 3


def test_replay_buffer_store_mask_skips_dummy_workers():
    buf = ReplayBuffer(100, {"obs": [2]}, 1)
    obs, act, rew, nxt, term, trunc = _batch(3)
    # worker 0 is a post-done autoreset dummy; only workers 1 and 2 are real.
    buf.add(
        {"obs": obs},
        act,
        rew,
        {"obs": nxt},
        term,
        trunc,
        store_mask=np.array([False, True, True]),
    )
    assert len(buf) == 2


def test_replay_buffer_all_dummy_mask_adds_nothing():
    buf = ReplayBuffer(100, {"obs": [2]}, 1)
    obs, act, rew, nxt, term, trunc = _batch(2)
    buf.add(
        {"obs": obs},
        act,
        rew,
        {"obs": nxt},
        term,
        trunc,
        store_mask=np.array([False, False]),
    )
    assert len(buf) == 0


def test_active_worker_indices_uses_sparse_store_mask_subset():
    assert list(_active_worker_indices(4, None)) == [0, 1, 2, 3]
    assert list(_active_worker_indices(4, np.array([True, True, True, True]))) == [0, 1, 2, 3]
    assert list(_active_worker_indices(4, np.array([False, True, False, True]))) == [1, 3]
    assert list(_active_worker_indices(4, np.array([False, False, False, False]))) == []


def test_active_worker_indices_rejects_mismatched_store_mask_length():
    with pytest.raises(ValueError, match="store_mask length"):
        list(_active_worker_indices(4, np.array([True, False])))


def test_nstep_multiworker_add_store_mask_skips_dummy_worker():
    buf = NstepReplayBuffer(100, {"obs": [2]}, 1, worker_size=2, n_step=2)
    obs, act, rew, nxt, term, trunc = _batch(2)
    for _ in range(2):
        buf.multiworker_add(
            {"obs": obs},
            act,
            rew,
            {"obs": nxt},
            term,
            trunc,
            store_mask=np.array([False, True]),
        )

    transitions = buf.get_buffer()
    assert len(buf) == 1
    assert np.array_equal(transitions["obs:obs"][0], obs[1])


# --- Fix 3: explicit --env_backend selection -----------------------------
def test_is_envpool_supported_true_for_canonical_env():
    assert eb._is_envpool_supported("CartPole-v1") is True


def test_is_envpool_supported_false_and_quiet_for_unknown_env(recwarn):
    # An unsupported env id is the expected outcome, not an error to shout about.
    assert eb._is_envpool_supported("NoSuchEnv-v999") is False
    assert len(recwarn) == 0


def _fake_vec(tag):
    def _make(env_id, worker_num, seed=None):
        return (tag, env_id, worker_num)

    return _make


def test_env_builder_default_backend_is_gymnasium(monkeypatch):
    monkeypatch.setattr(eb, "GymVectorizedEnv", _fake_vec("gym"))
    monkeypatch.setattr(eb, "EnvPoolVectorizedEnv", _fake_vec("envpool"))
    builder, _ = eb.get_env_builder("CartPole-v1")  # default backend
    assert builder(worker=4, seed=0) == ("gym", "CartPole-v1", 4)


def test_env_builder_envpool_backend_uses_envpool(monkeypatch):
    monkeypatch.setattr(eb, "GymVectorizedEnv", _fake_vec("gym"))
    monkeypatch.setattr(eb, "EnvPoolVectorizedEnv", _fake_vec("envpool"))
    builder, _ = eb.get_env_builder("CartPole-v1", env_backend="envpool")
    assert builder(worker=4, seed=0) == ("envpool", "CartPole-v1", 4)


def test_env_builder_envpool_backend_unsupported_env_raises():
    builder, _ = eb.get_env_builder("NoSuchEnv-v999", env_backend="envpool")
    with pytest.raises(ValueError, match="EnvPool has no spec"):
        builder(worker=4, seed=0)


@pytest.mark.parametrize(
    "env_id, expected",
    [("Pong-v5", True), ("CartPole-v1", False)],
)
def test_envpool_atari_detection_uses_env_spec_type(env_id, expected):
    env = eb.EnvPoolVectorizedEnv.__new__(eb.EnvPoolVectorizedEnv)
    assert env._check_atari_env(env_id) is expected


def test_get_env_builder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="env_backend must be"):
        eb.get_env_builder("CartPole-v1", env_backend="nope")


def _fake_recv_env(is_atari):
    """An EnvPoolVectorizedEnv whose ``self.env.recv()`` yields one fixed batch.

    Mirrors the real async contract: ``get_result()`` pulls via ``recv()`` and
    re-sorts by ``info['env_id']``, so the fake must supply that field.
    """
    env = eb.EnvPoolVectorizedEnv.__new__(eb.EnvPoolVectorizedEnv)
    env._is_atari = is_atari
    env.worker_num = 2
    env.obs = None
    env._awaiting_recv = True
    next_obs = np.zeros((2, 4), dtype=np.float32)
    rewards = np.array([1.0, -1.0], dtype=np.float32)
    terminateds = np.array([False, True])
    truncateds = np.array([False, False])
    infos = {
        "reward": np.array([4.0, -2.0], dtype=np.float32),
        "env_id": np.array([0, 1], dtype=np.int32),
    }
    env.env = SimpleNamespace(recv=lambda: (next_obs, rewards, terminateds, truncateds, infos))
    return env, infos


def test_envpool_atari_exposes_info_reward_as_original_reward():
    env, infos = _fake_recv_env(is_atari=True)

    _, _, _, _, result_infos = env.get_result()

    assert np.array_equal(result_infos["original_reward"], infos["reward"])
    assert "original_reward" not in infos


def test_envpool_non_atari_does_not_alias_info_reward():
    env, _ = _fake_recv_env(is_atari=False)

    _, _, _, _, result_infos = env.get_result()

    assert "original_reward" not in result_infos


def test_envpool_atari_real_reset_mask_excludes_lifeloss():
    env = eb.EnvPoolVectorizedEnv.__new__(eb.EnvPoolVectorizedEnv)
    env._is_atari = True
    env.worker_num = 3

    mask = env.real_reset_mask(
        terminateds=np.array([True, True, False]),
        truncateds=np.array([False, False, True]),
        infos={"lives": np.array([2, 0, 2], dtype=np.int32)},
    )

    assert mask.tolist() == [False, True, True]


def test_envpool_atari_autoreset_mask_excludes_lifeloss():
    env = eb.EnvPoolVectorizedEnv.__new__(eb.EnvPoolVectorizedEnv)
    env._is_atari = True
    env.worker_num = 3

    mask = env.autoreset_mask(
        terminateds=np.array([True, True, False]),
        truncateds=np.array([False, False, True]),
        infos={"lives": np.array([2, 0, 2], dtype=np.int32)},
    )

    assert mask.tolist() == [False, True, True]


def test_gym_atari_autoreset_mask_keeps_lifeloss_dummy_separate_from_real_reset():
    env = eb.GymVectorizedEnv.__new__(eb.GymVectorizedEnv)
    env._is_atari = True

    terminateds = np.array([True, True, False])
    truncateds = np.array([False, False, True])
    infos = {"lives": np.array([2, 0, 2], dtype=np.int32)}

    assert env.real_reset_mask(terminateds, truncateds, infos).tolist() == [False, True, True]
    assert env.autoreset_mask(terminateds, truncateds, infos).tolist() == [True, True, True]


def test_envpool_get_result_resorts_recv_into_canonical_env_order():
    # recv() may hand envs back in completion order; get_result() must re-sort
    # every per-env array by info["env_id"] so worker i always occupies row i --
    # the rollout loops index scores/prev_done/replay rows by that position.
    env = eb.EnvPoolVectorizedEnv.__new__(eb.EnvPoolVectorizedEnv)
    env._is_atari = False
    env.worker_num = 3
    env.obs = None
    env._awaiting_recv = True
    # Row r belongs to env env_id[r]; values encode the owning env so the
    # canonical re-sort is observable. This batch arrives as env 2, 0, 1.
    env_id = np.array([2, 0, 1], dtype=np.int32)
    next_obs = np.array([[2.0], [0.0], [1.0]], dtype=np.float32)
    rewards = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    terminateds = np.array([True, False, False])  # the terminal belongs to env2
    truncateds = np.array([False, False, False])
    infos = {"reward": np.array([20.0, 0.0, 10.0], dtype=np.float32), "env_id": env_id}
    env.env = SimpleNamespace(recv=lambda: (next_obs, rewards, terminateds, truncateds, infos))

    obs, rew, term, _, result_infos = env.get_result()

    # Canonical 0..N-1 order: env0, env1, env2 -- every array re-sorted in lockstep.
    assert obs["obs"].tolist() == [[0.0], [1.0], [2.0]]
    assert rew.tolist() == [0.0, 1.0, 2.0]
    assert term.tolist() == [False, False, True]
    assert result_infos["reward"].tolist() == [0.0, 10.0, 20.0]
    assert result_infos["env_id"].tolist() == [0, 1, 2]
