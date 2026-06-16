"""Regression tests for on-policy episode-boundary handling.

Two coupled fixes (2026-06-14):

- ``discount_with_terminated`` (A2C return) used to reset its backward
  accumulation only on ``truncated``, so a ``terminated`` episode end
  (gymnasium/EnvPool natural termination: term=1, trunc=0) did NOT reset and the
  return bled across episodes. It now resets on the boundary ``done=term|trunc``.
- The shared A2C ``learn_VectorizedEnv`` loop now flags the autoreset *dummy*
  step (the post-``done`` step EnvPool/gymnasium vector envs emit with the action
  ignored and reward 0) as terminal, so it yields a zero-value target instead of
  bridging two episodes.
"""

from collections import deque

import jax.numpy as jnp
import numpy as np
import pytest

from env_builder import env_builder as eb
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.returns import discount_with_terminated, get_vtrace

GAMMA = 0.9


def _returns(rewards, terminateds, truncateds, next_values):
    def col(x):
        return jnp.array(x, dtype=jnp.float32).reshape(-1, 1)

    out = discount_with_terminated(
        col(rewards), col(terminateds), col(truncateds), col(next_values), GAMMA
    )
    return [round(float(v), 3) for v in np.asarray(out).reshape(-1)]


def test_discount_resets_accumulation_on_terminated():
    # rewards all 1, terminate at idx2 (term=1, trunc=0). The return must reset
    # at the terminal instead of carrying the forced-last-step bootstrap back.
    assert _returns([1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [10, 10, 10, 10]) == [
        2.71,
        1.9,
        1.0,
        10.0,
    ]


def test_discount_bootstraps_on_truncation_not_termination():
    # Non-regression guard: truncation-bootstrap was correct both before and
    # after the fix; it pins that behavior, it does not exercise the bleed fix.
    # idx2 truncated (not terminal) -> bootstrap from next_value (10).
    trunc_returns = _returns([1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0], [10, 10, 10, 10])
    # idx2 terminated -> no bootstrap (return is just its own reward).
    term_returns = _returns([1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [10, 10, 10, 10])
    assert trunc_returns[2] == round(1 + GAMMA * 10, 3)  # 10.0
    assert term_returns[2] == 1.0


def test_discount_does_not_bleed_across_episode_boundary():
    # EnvPool autoreset pattern: real terminal at idx1 (term=1), dummy at idx2.
    # Episode A (idx0, idx1) must be clean -- no reward from the next episode.
    out = _returns([1, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0], [10, 10, 10, 10])
    assert out[0] == 1.9 and out[1] == 1.0  # episode A isolated


# --- V-trace estimator: shared by every IMPALA variant (A2C/PPO/TPPO/SPO) ---
def _vtrace(rewards, rhos, c_ts, terminateds, truncateds, values, next_values):
    def col(x):
        return jnp.array(x, dtype=jnp.float32).reshape(-1, 1)

    out = get_vtrace(
        col(rewards),
        col(rhos),
        col(c_ts),
        col(terminateds),
        col(truncateds),
        col(values),
        col(next_values),
        GAMMA,
    )
    return [round(float(v), 4) for v in np.asarray(out).reshape(-1)]


def test_vtrace_terminal_stops_bootstrap():
    # On-policy ratios (rho=1, c=0.95). Terminal at idx2 must zero the bootstrap
    # so vs[2] is just its own reward, not r + gamma*next_value.
    on_policy = dict(rhos=[1, 1, 1, 1], c_ts=[0.95, 0.95, 0.95, 0.95])
    vs = _vtrace(
        [1, 1, 1, 1],
        terminateds=[0, 0, 1, 0],
        truncateds=[0, 0, 0, 0],
        values=[0, 0, 0, 0],
        next_values=[5, 5, 5, 5],
        **on_policy,
    )
    assert vs == [10.9335, 6.355, 1.0, 5.5]


def test_vtrace_truncation_keeps_bootstrap():
    # Truncation (term=0) still bootstraps from next_value at idx2, unlike a
    # terminal -- pins the term/trunc asymmetry the V-trace correction relies on.
    on_policy = dict(rhos=[1, 1, 1, 1], c_ts=[0.95, 0.95, 0.95, 0.95])
    vs = _vtrace(
        [1, 1, 1, 1],
        terminateds=[0, 0, 0, 0],
        truncateds=[0, 0, 1, 0],
        values=[0, 0, 0, 0],
        next_values=[5, 5, 5, 5],
        **on_policy,
    )
    assert vs == [14.2231, 10.2025, 5.5, 5.5]


def test_vtrace_off_policy_ratio_scales_correction():
    # Clipped importance ratio rho=0.5 scales every per-step delta -- guards the
    # rho/c_t weighting against drift when the consolidated helper is refactored.
    vs = _vtrace(
        [1, 1, 1, 1],
        rhos=[0.5, 0.5, 0.5, 0.5],
        c_ts=[0.475, 0.475, 0.475, 0.475],
        terminateds=[0, 0, 0, 0],
        truncateds=[0, 0, 0, 0],
        values=[0, 0, 0, 0],
        next_values=[5, 5, 5, 5],
    )
    assert vs == [4.6431, 4.4282, 3.9256, 2.75]


# --- A2C vectorized loop: dummy-step neutralization -----------------------
class _ScriptEnv:
    ws = 2

    def __init__(self, term_script):
        self._script = term_script
        self._t = 0

    def current_obs(self):
        return np.zeros((self.ws, 2), dtype=np.float32)

    def step(self, actions):
        pass

    def get_result(self):
        terms = self._script[self._t]
        self._t += 1
        nxt = np.ones((self.ws, 2), dtype=np.float32)
        # Nonzero so the dummy-step reward zeroing is observable in the buffer.
        rewards = np.ones(self.ws, dtype=np.float32)
        truncs = np.zeros(self.ws, dtype=bool)
        return nxt, rewards, terms, truncs, {}


class _RecordingBuffer:
    def __init__(self):
        self.terminateds = []
        self.rewards = []
        self.truncateds = []

    def add(self, obs, action, reward, nxtobs, terminated, truncated):
        self.terminateds.append(np.asarray(terminated).copy())
        self.rewards.append(np.asarray(reward).copy())
        self.truncateds.append(np.asarray(truncated).copy())


class _Pbar(list):
    def set_description(self, _):
        pass


class _NullLogger:
    def log_metric(self, key, value, step):
        pass


class _Ctx:
    def __init__(self, steps):
        self.pbar = _Pbar(steps)
        self.eval_freq = 10_000
        self.log_interval = 10_000
        self.logger_run = _NullLogger()


def test_a2c_vectorized_flags_autoreset_dummy_step_as_terminal():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    # step0 normal, step1 worker0 terminates, step2 worker0 emits autoreset dummy.
    agent.env = _ScriptEnv(
        [
            np.array([False, False]),
            np.array([True, False]),
            np.array([False, False]),
        ]
    )
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 10_000  # train cadence never fires
    agent.lossque = deque(maxlen=10)
    agent.actions = lambda obs: np.zeros((2, 1), dtype=np.float32)
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([0, 1, 2]))

    stored_term = [t.tolist() for t in agent.buffer.terminateds]
    stored_rew = [r.tolist() for r in agent.buffer.rewards]
    stored_trunc = [t.tolist() for t in agent.buffer.truncateds]
    # step2: worker0 was done last step, so its dummy step is fully neutralized --
    # flagged terminal AND reward zeroed -- while worker1 (never done) and the
    # truncated flags are untouched. The env reported terminated [False, False]
    # and reward [1, 1] at step2.
    assert stored_term == [[False, False], [True, False], [True, False]]
    assert stored_rew == [[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert stored_trunc == [[False, False], [False, False], [False, False]]


def test_pipelined_loop_drives_real_async_envpool_end_to_end():
    # The on-policy loop now pipelines send(step)/recv(get_result) around a REAL
    # async EnvPool env. This exercises that composition end-to-end -- the loop's
    # current_obs -> step -> get_result alternation against the live send/recv
    # contract -- without touching the (separately broken) JAX return math: both
    # actions and train_step are stubbed. A crash here means the async handshake
    # or the buffer/episode bookkeeping drifted, not the learning update.
    pytest.importorskip("envpool")

    worker = 4
    env = eb.EnvPoolVectorizedEnv("CartPole-v1", worker_num=worker, seed=0)
    try:
        agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
        agent.env = env
        agent.worker_size = worker
        agent.batch_size = 2
        agent.lossque = deque(maxlen=10)
        trained = []
        agent.actions = lambda obs: np.zeros(worker, dtype=np.int32)
        agent.train_step = lambda steps: (trained.append(steps), 0.0)[1]
        agent.eval = lambda ctx, steps: None
        agent.description = lambda eval_result: "desc"
        agent.buffer = _RecordingBuffer()

        n_steps = 16
        Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx(list(range(n_steps))))
    finally:
        env.close()

    # One buffer.add per step, each a per-worker batch -- proves the recv results
    # flowed through the pipeline with the worker dimension intact.
    assert len(agent.buffer.rewards) == n_steps
    assert all(np.asarray(r).shape == (worker,) for r in agent.buffer.rewards)
    # The periodic train_step (the work that overlaps env stepping) actually ran.
    assert len(trained) >= 1
