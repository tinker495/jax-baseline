"""Tests for the ``rollout/`` measurement path (zero coverage before this).

Two layers:

- :class:`jax_baselines.common.rollout_stats.EpisodeTracker` windowing: the
  window mean is logged under the ``rollout/`` prefix, throttled to
  ``log_interval`` and skipping an empty window, with ``original_reward`` only
  when supplied.
- The :class:`jax_baselines.common.rollout.RolloutEngine` episode-end emit: the
  four loops hand each *completed* training episode to the
  ``record_rollout_episode`` callback, and the autoreset dummy step is never
  counted as an episode end (the ``active``/``prev_done`` mask).
"""

import numpy as np

from jax_baselines.common.rollout import ActionSelection, RolloutEngine, RolloutSpec
from jax_baselines.common.rollout_stats import EpisodeTracker


# --- EpisodeTracker -------------------------------------------------------
class _RecordingLog:
    def __init__(self):
        self.calls = []

    def __call__(self, key, value, step):
        self.calls.append((key, round(float(value), 4), step))

    def keyed(self, key):
        return [(c[1], c[2]) for c in self.calls if c[0] == key]


def test_tracker_logs_window_mean_under_rollout_prefix():
    log = _RecordingLog()
    tracker = EpisodeTracker(log, log_interval=100)

    # First episode is under the interval: accumulates, does not flush yet.
    tracker.record(50, episode_reward=1.0, episode_length=4, timeout=0.0)
    assert log.calls == []

    # Second episode crosses the interval boundary: flush window mean at step 150.
    tracker.record(150, episode_reward=3.0, episode_length=6, timeout=1.0)
    assert log.keyed("rollout/episode_reward") == [(2.0, 150)]
    assert log.keyed("rollout/episode_length") == [(5.0, 150)]
    assert log.keyed("rollout/timeout_rate") == [(0.5, 150)]


def test_tracker_throttles_to_log_interval():
    log = _RecordingLog()
    tracker = EpisodeTracker(log, log_interval=100)

    tracker.record(150, episode_reward=1.0, episode_length=1, timeout=0.0)  # flush @150
    tracker.record(160, episode_reward=2.0, episode_length=1, timeout=0.0)  # within interval
    tracker.record(170, episode_reward=3.0, episode_length=1, timeout=0.0)  # within interval
    tracker.record(260, episode_reward=4.0, episode_length=1, timeout=0.0)  # flush @260

    steps_logged = [step for _, step in log.keyed("rollout/episode_reward")]
    assert steps_logged == [150, 260]
    # The @260 flush means the last 4 rewards (window not yet full): mean(1,2,3,4)=2.5
    assert log.keyed("rollout/episode_reward")[-1] == (2.5, 260)


def test_tracker_windows_last_k_only():
    log = _RecordingLog()
    tracker = EpisodeTracker(log, log_interval=1, window=3)
    for i, reward in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
        tracker.record(i, episode_reward=reward, episode_length=1, timeout=0.0)
    # window=3 keeps the last three rewards (20,30,40) -> mean 30 at the last flush.
    assert log.keyed("rollout/episode_reward")[-1] == (30.0, 4)


def test_tracker_original_reward_only_when_present():
    log = _RecordingLog()
    tracker = EpisodeTracker(log, log_interval=1)
    tracker.record(1, episode_reward=1.0, episode_length=1, timeout=0.0)
    assert log.keyed("rollout/original_reward") == []

    tracker.record(2, episode_reward=1.0, episode_length=1, timeout=0.0, original_reward=42.0)
    assert log.keyed("rollout/original_reward") == [(42.0, 2)]


def test_tracker_describe_is_empty_until_first_episode():
    log = _RecordingLog()
    tracker = EpisodeTracker(log, log_interval=100)
    assert tracker.describe() == ""
    tracker.record(1, episode_reward=7.0, episode_length=1, timeout=0.0)
    assert "rollout_rew" in tracker.describe()


# --- RolloutEngine episode emit ------------------------------------------
class _ScriptedSingleEnv:
    ws = 1

    def __init__(self, script, infos=None):
        self.script = script  # list of (reward, terminated, truncated)
        self.infos = infos if infos is not None else [{}] * len(script)
        self.t = 0

    def reset(self):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        reward, terminated, truncated = self.script[self.t]
        info = self.infos[self.t]
        self.t += 1
        return np.zeros(1, dtype=np.float32), reward, terminated, truncated, info


class _ScriptedVecEnv:
    def __init__(self, script, worker_size):
        self.script = script  # list of (rewards, terminateds, truncateds)
        self.ws = worker_size
        self.i = 0

    def current_obs(self):
        return np.zeros((self.ws, 1), dtype=np.float32)

    def step(self, actions):
        pass

    def get_result(self):
        rewards, terminateds, truncateds = self.script[self.i]
        self.i += 1
        return np.zeros((self.ws, 1), dtype=np.float32), rewards, terminateds, truncateds, {}


class _NoopBuffer:
    def add(self, *args, **kwargs):
        pass


def _episode_recorder():
    records = []

    def record(steps, *, episode_reward, episode_length, timeout, original_reward=None):
        records.append(
            {
                "steps": steps,
                "reward": round(float(episode_reward), 4),
                "length": int(episode_length),
                "timeout": float(timeout),
                "original": original_reward,
            }
        )

    return records, record


def _spec(env, record, **overrides):
    """Minimal RolloutSpec whose train/eval cadence never fires, isolating the
    episode-record emit."""
    base = dict(
        env=env,
        replay_buffer=_NoopBuffer(),
        learning_starts=10**9,
        train_freq=1,
        gradient_steps=1,
        eval_freq=10**9,
        worker_size=env.ws,
        single_action=lambda obs, steps: ActionSelection(0, 0),
        vector_action=lambda obs, steps: ActionSelection(0, 0),
        refresh_exploration=lambda steps: None,
        has_true_reset=lambda: False,
        train=lambda steps, gs: 0.0,
        evaluate=lambda steps: None,
        describe=lambda eval_result: "desc",
        bind_loss_window=lambda window: None,
        record_rollout_episode=record,
        checkpoint_on_episode_end=lambda *a, **k: True,
        checkpoint_pulse=lambda *a, **k: None,
    )
    base.update(overrides)
    return RolloutSpec(**base)


def test_single_env_emits_completed_episode_records():
    # Two episodes: terminate at step1 (return 2, len 2), truncate at step3 (return 2, len 2).
    env = _ScriptedSingleEnv(
        [(1.0, False, False), (1.0, True, False), (1.0, False, False), (1.0, False, True)]
    )
    records, record = _episode_recorder()
    RolloutEngine(_spec(env, record)).learn_single_env(range(4), None, log_interval=10**9)

    assert [(r["reward"], r["length"], r["timeout"]) for r in records] == [
        (2.0, 2, 0.0),
        (2.0, 2, 1.0),
    ]


def test_single_env_accumulates_original_reward():
    env = _ScriptedSingleEnv(
        [(1.0, False, False), (1.0, True, False)],
        infos=[{"original_reward": 10.0}, {"original_reward": 20.0}],
    )
    records, record = _episode_recorder()
    RolloutEngine(_spec(env, record)).learn_single_env(range(2), None, log_interval=10**9)

    assert len(records) == 1
    assert records[0]["original"] == 30.0


def test_vectorized_env_excludes_autoreset_dummy_step():
    ws = 1

    def row(reward, term, trunc):
        return (
            np.array([reward], dtype=np.float32),
            np.array([term]),
            np.array([trunc]),
        )

    # step0 normal, step1 real terminate, step2 autoreset dummy (reward 5 must NOT
    # count), step3 truncate.
    env = _ScriptedVecEnv(
        [
            row(1.0, False, False),
            row(1.0, True, False),
            row(5.0, False, False),
            row(1.0, False, True),
        ],
        worker_size=ws,
    )
    records, record = _episode_recorder()
    RolloutEngine(_spec(env, record)).learn_vectorized_env(range(4), None, log_interval=10**9)

    # The dummy step's reward 5 is excluded; two clean episodes recorded.
    assert [(r["reward"], r["length"], r["timeout"]) for r in records] == [
        (2.0, 2, 0.0),
        (1.0, 1, 1.0),
    ]


def test_single_env_checkpointing_emits_after_learning_starts():
    # learning_starts=0 so both episode ends (steps 1 and 3, both > 0) emit.
    env = _ScriptedSingleEnv(
        [(1.0, False, False), (1.0, True, False), (1.0, False, False), (1.0, False, True)]
    )
    records, record = _episode_recorder()
    RolloutEngine(_spec(env, record, learning_starts=0)).learn_single_env_checkpointing(
        range(4), None, log_interval=10**9
    )

    assert [(r["reward"], r["length"], r["timeout"]) for r in records] == [
        (2.0, 2, 0.0),
        (2.0, 2, 1.0),
    ]


def test_vectorized_checkpointing_excludes_dummy_and_guards_active():
    ws = 1

    def row(reward, term, trunc):
        return (
            np.array([reward], dtype=np.float32),
            np.array([term]),
            np.array([trunc]),
        )

    # step1 real terminate, step2 autoreset dummy (reward 5 must NOT count), step3 truncate.
    env = _ScriptedVecEnv(
        [
            row(1.0, False, False),
            row(1.0, True, False),
            row(5.0, False, False),
            row(1.0, False, True),
        ],
        worker_size=ws,
    )
    records, record = _episode_recorder()
    RolloutEngine(_spec(env, record, learning_starts=0)).learn_vectorized_env_checkpointing(
        range(4), None, log_interval=10**9
    )

    assert [(r["reward"], r["length"], r["timeout"]) for r in records] == [
        (2.0, 2, 0.0),
        (1.0, 1, 1.0),
    ]
