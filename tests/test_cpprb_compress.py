"""cpprb image/frame-stack compression (compress_memory) correctness.

Strategy: feed an identical scripted sequence of frame-stacked transitions to a
compress-enabled buffer and an uncompressed reference buffer, then assert their
stored transitions (reconstructed by cpprb) are element-wise identical. cpprb's
``next_of`` / ``stack_compress`` are lossless, so any divergence is a wrapper bug.
"""

import numpy as np
import pytest

from jax_baselines.common.cpprb_buffers import (
    MultiPrioritizedReplayBuffer,
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from jax_baselines.common.replay_factory import make_replay_buffer

H = W = 8
S = 4  # frame-stack depth, as in Atari
IMG = [H, W, S]
OBS = [IMG]  # observation_space is a list of per-modality shapes


def _frame(v):
    return np.full((H, W), v % 256, dtype=np.uint8)


def _stack(vals):
    """Frame-stack along the last axis (newest frame last), batch dim of 1."""
    return np.stack([_frame(v) for v in vals], axis=-1)[np.newaxis, ...]


def _episode(base, length):
    """Yield (obs, next_obs) frame-stacked windows for one episode.

    Mirrors gym FrameStack: the first observation repeats its initial frame, and
    each subsequent observation slides the window by one frame.
    """
    frames = [base + f for f in range(length + 1)]
    for f in range(length):
        win = [frames[max(0, f - S + 1 + i)] for i in range(S)]
        nxt = [frames[max(0, f - S + 2 + i)] for i in range(S)]
        yield _stack(win), _stack(nxt)


EPISODES = [(10, 5), (100, 7), (200, 4)]  # disjoint frame ranges -> bleed is visible


def _feed_single(buf):
    for base, length in EPISODES:
        for step, (obs, nxt) in enumerate(_episode(base, length)):
            terminated = step == length - 1
            buf.add([obs], 1, 1.0, [nxt], terminated, False)


def _assert_lossless(compressed, reference):
    c = compressed.get_buffer()
    r = reference.get_buffer()
    assert np.array_equal(c["obs0"], r["obs0"]), "obs0 reconstruction diverged"
    assert np.array_equal(c["next_obs0"], r["next_obs0"]), "next_obs0 reconstruction diverged"


def test_replay_buffer_compress_lossless():
    comp = ReplayBuffer(500, OBS, action_space=1, compress_memory=True)
    ref = ReplayBuffer(500, OBS, action_space=1, compress_memory=False)
    _feed_single(comp)
    _feed_single(ref)
    assert comp._compress_active is True
    _assert_lossless(comp, ref)
    # sample returns both obs and next_obs for the image modality
    smpl = comp.sample(16)
    assert smpl["obses"][0].shape == (16, H, W, S)
    assert smpl["nxtobses"][0].shape == (16, H, W, S)


def test_prioritized_buffer_compress_lossless():
    comp = PrioritizedReplayBuffer(500, OBS, alpha=0.6, action_space=1, compress_memory=True)
    ref = PrioritizedReplayBuffer(500, OBS, alpha=0.6, action_space=1, compress_memory=False)
    _feed_single(comp)
    _feed_single(ref)
    assert comp._compress_active is True
    _assert_lossless(comp, ref)
    smpl = comp.sample(16, 0.4)
    assert smpl["nxtobses"][0].shape == (16, H, W, S)
    assert "weights" in smpl and "indexes" in smpl


@pytest.mark.parametrize("cls", [NstepReplayBuffer, PrioritizedNstepReplayBuffer])
def test_nstep_buffer_compress_lossless(cls):
    kw = dict(action_space=1, worker_size=1, n_step=3, gamma=0.99)
    comp = cls(500, OBS, compress_memory=True, **kw)
    ref = cls(500, OBS, compress_memory=False, **kw)
    _feed_single(comp)
    _feed_single(ref)
    _assert_lossless(comp, ref)


def test_nstep_multiworker_compress_lossless():
    workers = 2
    comp = NstepReplayBuffer(
        500, OBS, action_space=1, worker_size=workers, n_step=3, gamma=0.99, compress_memory=True
    )
    ref = NstepReplayBuffer(
        500, OBS, action_space=1, worker_size=workers, n_step=3, gamma=0.99, compress_memory=False
    )
    # Two workers stepping in lockstep; each ends its episode at a different time.
    gens = [[list(_episode(b, length)) for b, length in EPISODES] for _ in range(workers)]
    for ep_idx, (base, length) in enumerate(EPISODES):
        for step in range(length):
            obs_w = np.concatenate([gens[w][ep_idx][step][0] for w in range(workers)], axis=0)
            nxt_w = np.concatenate([gens[w][ep_idx][step][1] for w in range(workers)], axis=0)
            term = np.array([step == length - 1] * workers)
            trunc = np.array([False] * workers)
            comp.add([obs_w], np.ones((workers, 1)), np.ones(workers), [nxt_w], term, trunc)
            ref.add([obs_w], np.ones((workers, 1)), np.ones(workers), [nxt_w], term, trunc)
    _assert_lossless(comp, ref)


def test_vector_obs_compress_is_noop():
    # No image modality -> compression must be inert and still round-trip.
    vec = [[4]]
    buf = ReplayBuffer(100, vec, action_space=1, compress_memory=True)
    assert buf._compress_active is False
    for t in range(10):
        buf.add(
            [np.arange(t, t + 4, dtype=np.float32)[np.newaxis, :]],
            1,
            1.0,
            [np.arange(t + 1, t + 5, dtype=np.float32)[np.newaxis, :]],
            t == 9,
            False,
        )
    smpl = buf.sample(5)
    assert smpl["obses"][0].shape == (5, 4)
    assert smpl["nxtobses"][0].shape == (5, 4)


@pytest.mark.parametrize("prioritized", [False, True])
def test_factory_forwards_compress_memory_single_step(prioritized):
    # Regression: make_replay_buffer used to drop compress_memory on every branch,
    # so --compress_memory was silently a no-op. Single-step image compress stays on
    # cpprb (next_of + stack_compress) and must round-trip losslessly.
    common = dict(worker_size=1, n_step=1, gamma=0.99, prioritized=prioritized, alpha=0.6, eps=1e-3)
    comp = make_replay_buffer(500, OBS, 1, compress_memory=True, **common)
    ref = make_replay_buffer(500, OBS, 1, compress_memory=False, **common)
    _feed_single(comp)
    _feed_single(ref)
    assert comp._compress_active is True
    _assert_lossless(comp, ref)


@pytest.mark.parametrize("prioritized", [False, True])
def test_factory_routes_nstep_image_compress_to_frame_buffer(prioritized):
    # n-step image + compress can't use cpprb (stack_compress breaks the n-step
    # next_obs), so the factory routes to the frame-level buffer instead.
    from jax_baselines.common.frame_buffers import (
        FrameStackReplayBuffer,
        PrioritizedFrameStackReplayBuffer,
    )

    buf = make_replay_buffer(
        500,
        OBS,
        1,
        worker_size=1,
        n_step=3,
        gamma=0.99,
        prioritized=prioritized,
        alpha=0.6,
        eps=1e-3,
        compress_memory=True,
    )
    expected = PrioritizedFrameStackReplayBuffer if prioritized else FrameStackReplayBuffer
    assert isinstance(buf, expected)


def test_multiprioritized_warns_and_keeps_next_obs():
    with pytest.warns(RuntimeWarning, match="compress_memory is unsupported"):
        mp = MultiPrioritizedReplayBuffer(
            64, OBS, 0.6, action_space=1, n_step=1, gamma=0.99, manager=None, compress_memory=True
        )
    # central buffer must store next_obs fully (no broken deletion)
    assert "next_obs0" in mp.env_dict
