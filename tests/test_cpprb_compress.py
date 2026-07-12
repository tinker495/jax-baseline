"""cpprb image/frame-stack compression (compress_memory) correctness.

Strategy: feed an identical scripted sequence of frame-stacked transitions to a
compress-enabled buffer and an uncompressed reference buffer, then assert their
stored transitions (reconstructed by cpprb) are element-wise identical. cpprb's
``next_of`` / ``stack_compress`` are lossless, so any divergence is a wrapper bug.
"""

import numpy as np
import pytest

from jax_baselines.core.replay_protocol import LocalReplayNeed, PriorityNeed
from replay_memory.cpprb_buffers import (
    MultiPrioritizedReplayBuffer,
    NstepReplayBuffer,
    PrioritizedNstepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    _project_transitions,
)
from replay_memory.replay_factory import make_replay_buffer

H = W = 8
S = 4  # frame-stack depth, as in Atari
IMG = [H, W, S]
OBS = [IMG]  # observation_space is a list of per-modality shapes


def test_transition_projection_has_one_plain_and_prioritized_shape():
    transitions = {
        "obs0": np.zeros((2, 4), dtype=np.float32),
        "action": np.zeros((2, 1), dtype=np.float32),
        "reward": np.zeros((2, 1), dtype=np.float32),
        "next_obs0": np.ones((2, 4), dtype=np.float32),
        "done": np.zeros((2, 1), dtype=np.float32),
        "weights": np.ones(2, dtype=np.float32),
        "indexes": np.arange(2),
    }

    plain = _project_transitions(transitions, ["obs0"], ["next_obs0"])
    prioritized = _project_transitions(transitions, ["obs0"], ["next_obs0"], prioritized=True)

    assert set(plain) == {"obses", "actions", "rewards", "nxtobses", "terminateds"}
    assert set(prioritized) == set(plain) | {"weights", "indexes"}
    assert plain["obses"] == [transitions["obs0"]]
    assert plain["nxtobses"] == [transitions["next_obs0"]]


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


def _need(
    *,
    buffer_size=500,
    observation_space=OBS,
    action_shape_or_n=1,
    worker_size=1,
    n_step=1,
    gamma=0.99,
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
        500,
        OBS,
        action_space=1,
        worker_size=workers,
        n_step=3,
        gamma=0.99,
        compress_memory=True,
    )
    ref = NstepReplayBuffer(
        500,
        OBS,
        action_space=1,
        worker_size=workers,
        n_step=3,
        gamma=0.99,
        compress_memory=False,
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
    common = dict(
        worker_size=1,
        n_step=1,
        gamma=0.99,
        prioritized=prioritized,
        alpha=0.6,
        eps=1e-3,
    )
    comp = make_replay_buffer(_need(compress_memory=True, **common))
    ref = make_replay_buffer(_need(compress_memory=False, **common))
    _feed_single(comp)
    _feed_single(ref)
    assert comp._compress_active is True
    _assert_lossless(comp, ref)


@pytest.mark.parametrize(
    "prioritized,expected",
    [
        (False, NstepReplayBuffer),
        (True, PrioritizedNstepReplayBuffer),
    ],
)
def test_factory_routes_multiworker_single_step_image_compress_to_staged_cpprb(
    prioritized, expected
):
    buf = make_replay_buffer(
        _need(worker_size=2, n_step=1, prioritized=prioritized, compress_memory=True)
    )

    assert isinstance(buf, expected)
    assert buf.worker_size == 2


@pytest.mark.parametrize("prioritized", [False, True])
def test_multiworker_single_step_image_compress_accepts_vector_done_arrays(prioritized):
    buf = make_replay_buffer(
        _need(worker_size=2, n_step=1, prioritized=prioritized, compress_memory=True)
    )

    obs = np.concatenate([_stack([10, 10, 10, 10]), _stack([100, 100, 100, 100])], axis=0)
    nxt = np.concatenate([_stack([10, 10, 10, 11]), _stack([100, 100, 100, 101])], axis=0)
    action = np.ones((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    no_done = np.array([False, False])

    buf.add([obs], action, reward, [nxt], no_done, no_done)
    buf.add([obs], action, reward, [nxt], np.array([True, False]), no_done)

    assert len(buf) == 4
    sample = buf.sample(1, 0.4) if prioritized else buf.sample(1)
    assert sample["obses"][0].shape == (1, H, W, S)
    assert sample["nxtobses"][0].shape == (1, H, W, S)


@pytest.mark.parametrize("prioritized", [False, True])
def test_multiworker_single_step_image_compress_is_available_before_done(prioritized):
    buf = make_replay_buffer(
        _need(worker_size=2, n_step=1, prioritized=prioritized, compress_memory=True)
    )

    obs = np.concatenate([_stack([10, 10, 10, 10]), _stack([100, 100, 100, 100])], axis=0)
    nxt = np.concatenate([_stack([10, 10, 10, 11]), _stack([100, 100, 100, 101])], axis=0)
    action = np.ones((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    no_done = np.array([False, False])

    buf.add([obs], action, reward, [nxt], no_done, no_done)

    assert len(buf) == 2
    sample = buf.sample(1, 0.4) if prioritized else buf.sample(1)
    assert sample["obses"][0].shape == (1, H, W, S)
    assert sample["nxtobses"][0].shape == (1, H, W, S)


@pytest.mark.parametrize("prioritized", [False, True])
def test_multiworker_single_step_image_compress_store_mask_is_lossless(prioritized):
    comp = make_replay_buffer(
        _need(worker_size=2, n_step=1, prioritized=prioritized, compress_memory=True)
    )
    ref = make_replay_buffer(
        _need(worker_size=2, n_step=1, prioritized=prioritized, compress_memory=False)
    )
    action = np.ones((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    no_done = np.array([False, False])

    for step, store_mask in enumerate(
        [
            np.array([True, True]),
            np.array([False, True]),
            np.array([True, True]),
        ]
    ):
        obs = np.concatenate(
            [
                _stack([10 + step, 10 + step, 10 + step, 10 + step]),
                _stack([100 + step, 100 + step, 100 + step, 100 + step]),
            ],
            axis=0,
        )
        nxt = np.concatenate(
            [
                _stack([10 + step, 10 + step, 10 + step, 11 + step]),
                _stack([100 + step, 100 + step, 100 + step, 101 + step]),
            ],
            axis=0,
        )
        comp.add([obs], action, reward, [nxt], no_done, no_done, store_mask=store_mask)
        ref.add([obs], action, reward, [nxt], no_done, no_done, store_mask=store_mask)

    _assert_lossless(comp, ref)


@pytest.mark.parametrize("prioritized", [False, True])
def test_multiworker_single_step_image_compress_does_not_cap_long_episode(prioritized):
    buf = make_replay_buffer(
        _need(
            buffer_size=2505,
            worker_size=2,
            n_step=1,
            prioritized=prioritized,
            compress_memory=True,
        )
    )
    obs = np.concatenate([_stack([10, 10, 10, 10]), _stack([100, 100, 100, 100])], axis=0)
    nxt = np.concatenate([_stack([10, 10, 10, 11]), _stack([100, 100, 100, 101])], axis=0)
    action = np.ones((2, 1), dtype=np.float32)
    reward = np.ones(2, dtype=np.float32)
    truncated = np.array([False, False])
    store_mask = np.array([True, False])

    for step in range(2501):
        terminated = np.array([step == 2500, False])
        buf.add([obs], action, reward, [nxt], terminated, truncated, store_mask=store_mask)

    assert len(buf) == 2501


@pytest.mark.parametrize("prioritized", [False, True])
def test_factory_routes_nstep_image_compress_to_frame_buffer(prioritized):
    # n-step image + compress can't use cpprb (stack_compress breaks the n-step
    # next_obs), so the factory routes to the frame-level buffer instead.
    from replay_memory.frame_buffers import (
        FrameStackReplayBuffer,
        PrioritizedFrameStackReplayBuffer,
    )

    buf = make_replay_buffer(
        _need(
            worker_size=1,
            n_step=3,
            gamma=0.99,
            prioritized=prioritized,
            alpha=0.6,
            eps=1e-3,
            compress_memory=True,
        )
    )
    expected = PrioritizedFrameStackReplayBuffer if prioritized else FrameStackReplayBuffer
    assert isinstance(buf, expected)


def test_multiprioritized_warns_and_keeps_next_obs():
    with pytest.warns(RuntimeWarning, match="compress_memory is unsupported"):
        mp = MultiPrioritizedReplayBuffer(
            64,
            OBS,
            0.6,
            action_space=1,
            n_step=1,
            gamma=0.99,
            manager=None,
            compress_memory=True,
        )
    # central buffer must store next_obs fully (no broken deletion)
    assert "next_obs0" in mp.env_dict
