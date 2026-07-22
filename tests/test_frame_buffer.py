"""FrameStackReplayBuffer correctness: n-step transitions must match cpprb's
NstepReplayBuffer (ground truth) while storing only one frame per observation."""

from collections import deque

import numpy as np
import pytest

from jax_baselines.core.bulk_training import iter_bulk_batches
from jax_baselines.core.replay_protocol import LocalReplayNeed, PriorityNeed
from jax_baselines.DQN.training import QNetTrainingLifecycle, QNetTrainResult
from replay_memory.cpprb_buffers import NstepReplayBuffer
from replay_memory.frame_buffers import (
    FrameStackReplayBuffer,
    PrioritizedFrameStackReplayBuffer,
)
from replay_memory.replay_factory import make_replay_buffer

H = W = 6
S = 4
NSTEP = 3
GAMMA = 0.99
OBS = {"obs": [H, W, S]}  # Cf = 1 (Atari grayscale frame stack)
EPISODES = [(10, 5), (40, 7), (80, 6)]


def _need(
    *,
    buffer_size=500,
    observation_space=OBS,
    action_shape_or_n=1,
    worker_size=1,
    n_step=1,
    gamma=GAMMA,
    prioritized=False,
    alpha=0.6,
    eps=1e-3,
    compress_memory=False,
    n_frames=S,
):
    priority = PriorityNeed(alpha=alpha, eps=eps) if prioritized else None
    return LocalReplayNeed(
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_shape_or_n=action_shape_or_n,
        worker_size=worker_size,
        n_step=n_step,
        gamma=gamma,
        priority=priority,
        compress_observations=compress_memory,
        n_frames=n_frames,
    )


def _frame(v):
    return np.full((H, W, 1), v % 256, dtype=np.uint8)


class _FrameStacker:
    """Mirror of gym FrameStack: repeat the reset frame, newest in last channel."""

    def __init__(self):
        self.d = deque(maxlen=S)

    def reset(self, v):
        for _ in range(S):
            self.d.append(_frame(v))
        return np.concatenate(list(self.d), axis=-1)

    def step(self, v):
        self.d.append(_frame(v))
        return np.concatenate(list(self.d), axis=-1)


def _feed(buf):
    for base, length in EPISODES:
        fs = _FrameStacker()
        obs = fs.reset(base)
        for t in range(length):
            nxt = fs.step(base + t + 1)
            buf.add(
                {"obs": obs[None]},
                np.float32(1.0),
                np.float32(base + t),
                {"obs": nxt[None]},
                t == length - 1,
                False,
            )
            obs = nxt


def _reference():
    ref = NstepReplayBuffer(1000, OBS, action_space=1, worker_size=1, n_step=NSTEP, gamma=GAMMA)
    _feed(ref)
    return ref.get_buffer()


def test_nstep_transitions_match_cpprb():
    fb = FrameStackReplayBuffer(1000, OBS, 1, NSTEP, GAMMA, n_frames=S)
    _feed(fb)
    ref = _reference()
    total = len(ref["obs:obs"])
    g = fb._gather(np.arange(0, total, dtype=np.int64))
    # obs stack reconstruction is lossless
    assert np.array_equal(g["obses"]["obs"], ref["obs:obs"])
    # n-step discounted reward and done match for every row
    assert np.allclose(g["rewards"][:, 0], ref["reward"][:, 0], atol=1e-4)
    assert np.array_equal(g["terminateds"][:, 0].astype(bool), ref["done"][:, 0].astype(bool))
    # next_obs matches wherever it is used (done == 0; done == 1 is masked in the target)
    mask = ref["done"][:, 0] == 0
    assert mask.sum() > 0
    assert np.array_equal(g["nxtobses"]["obs"][mask], ref["next_obs:obs"][mask])


def test_single_frame_storage():
    fb = FrameStackReplayBuffer(1000, OBS, 1, NSTEP, GAMMA, n_frames=S)
    # one frame (Cf=1) per slot, not the full S-frame stack
    assert fb._frame.shape == (1000, H, W, 1)


def test_prioritized_sample_consistent_with_ground_truth():
    plain = FrameStackReplayBuffer(1000, OBS, 1, NSTEP, GAMMA, n_frames=S)
    pri = PrioritizedFrameStackReplayBuffer(1000, OBS, 1, NSTEP, GAMMA, alpha=0.6, n_frames=S)
    _feed(plain)
    _feed(pri)
    ground = plain._gather(np.arange(0, plain._ready(), dtype=np.int64))
    s = pri.sample(64, beta=0.4)
    leaves = s["indexes"]
    assert (leaves < pri._ready()).all()
    for i, leaf in enumerate(leaves):
        assert np.array_equal(s["obses"]["obs"][i], ground["obses"]["obs"][leaf])
        assert np.isclose(s["rewards"][i, 0], ground["rewards"][leaf, 0])
        assert np.array_equal(s["nxtobses"]["obs"][i], ground["nxtobses"]["obs"][leaf])
    assert (s["weights"] > 0).all() and (s["weights"] <= 1.0 + 1e-6).all()
    pri.update_priorities(leaves, np.abs(np.random.randn(64)).astype(np.float32) + 0.1)
    assert pri.sample(16)["obses"]["obs"].shape == (16, H, W, S)


class _FrameBufferBulkAgent:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.batch_size = 4
        self.max_bulk_updates_per_pulse = 2
        self.prioritized_replay = True
        self.prioritized_replay_beta0 = 0.4
        self.train_steps_count = 0
        self.logger_run = None
        self._last_log_step = 0
        self.log_interval = 1_000_000
        self.reward_normalizer = None

    def _sample_batch(self, batch_size=None):
        return self.replay_buffer.sample(self.batch_size if batch_size is None else batch_size)

    def _train_on_batch(self, data, context):
        return QNetTrainResult.from_values(
            loss=float(context.train_steps_count),
            replay_priorities=np.repeat(context.train_steps_count, self.batch_size),
        )

    def _train_on_bulk(self, data, contexts):
        results = [
            self._train_on_batch(batch, context)
            for batch, context in zip(iter_bulk_batches(data, contexts), contexts)
        ]
        report = self._aggregate_train_reports([result.report for result in results])
        report.update_count = sum(result.report.update_count for result in results)
        priorities = np.stack([result.replay_priorities for result in results])
        return QNetTrainResult(report=report, replay_priorities=priorities)

    def _aggregate_train_reports(self, reports):
        return reports[-1]


def test_qnet_bulk_priority_updates_flatten_for_prioritized_frame_buffer():
    pri = PrioritizedFrameStackReplayBuffer(1000, OBS, 1, NSTEP, GAMMA, alpha=0.6, n_frames=S)
    _feed(pri)
    agent = _FrameBufferBulkAgent(pri)

    loss = QNetTrainingLifecycle(agent).train(steps=10, gradient_steps=2)

    assert loss == 2.0
    assert agent.train_steps_count == 2
    assert pri.sample(4)["obses"]["obs"].shape == (4, H, W, S)


def test_truncation_bootstraps_without_crash():
    fb = FrameStackReplayBuffer(200, OBS, 1, NSTEP, GAMMA, n_frames=S)
    fs = _FrameStacker()
    obs = fs.reset(0)
    for t in range(6):
        nxt = fs.step(t + 1)
        fb.add(
            {"obs": obs[None]},
            np.float32(1),
            np.float32(1),
            {"obs": nxt[None]},
            False,
            t == 5,
        )
        obs = nxt
    fs2 = _FrameStacker()
    obs = fs2.reset(50)
    for t in range(6):
        nxt = fs2.step(51 + t)
        fb.add(
            {"obs": obs[None]},
            np.float32(1),
            np.float32(1),
            {"obs": nxt[None]},
            False,
            False,
        )
        obs = nxt
    g = fb._gather(np.arange(0, fb._ready(), dtype=np.int64))
    assert (g["terminateds"] == 0).all()  # truncation does not set done
    assert g["obses"]["obs"].shape[1:] == (H, W, S)


@pytest.mark.parametrize("prioritized", [False, True])
def test_factory_selects_frame_buffer_for_image_nstep_compress(prioritized):
    buf = make_replay_buffer(
        _need(
            n_step=NSTEP,
            gamma=GAMMA,
            prioritized=prioritized,
            compress_memory=True,
            n_frames=S,
        )
    )
    expected = PrioritizedFrameStackReplayBuffer if prioritized else FrameStackReplayBuffer
    assert isinstance(buf, expected)


def test_factory_keeps_cpprb_for_vector_or_singlestep():
    from replay_memory.cpprb_buffers import NstepReplayBuffer as Cpprb

    # vector obs -> not frame-compressible
    assert isinstance(
        make_replay_buffer(
            _need(observation_space={"obs": [4]}, n_step=NSTEP, compress_memory=True)
        ),
        Cpprb,
    )
    # single-step image -> cpprb next_of+stack_compress path, not the frame buffer
    assert not isinstance(
        make_replay_buffer(_need(n_step=1, compress_memory=True)),
        FrameStackReplayBuffer,
    )
