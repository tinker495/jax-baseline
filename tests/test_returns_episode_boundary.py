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

import ast
import inspect
import textwrap
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env_builder import env_builder as eb
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.APE_X.dpg_worker import Ape_X_Worker as ApeXDPGWorker
from jax_baselines.APE_X.worker import Ape_X_Worker as ApeXQWorker
from jax_baselines.IMPALA.worker import Impala_Worker
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


def test_discount_with_terminated_real_train_step_contract():
    # The [T, 1]-float tests above mask the real bug: A2C's _train_step feeds
    # discount_with_terminated through jax.vmap with the genuine buffer layout --
    # rewards/terminateds/truncateds are FLAT [worker, T] and the done-flags are
    # BOOL, while next_values keeps the critic's trailing unit dim [worker, T, 1].
    # Before the fix this raised (truncateds[-1] scalar broadcast, then bool
    # subtraction in `term + trunc - term*trunc`). Pins that contract so the
    # regression that silently broke `pg --algo A2C` cannot return.
    rewards = jnp.array([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)  # [worker=1, T=4]
    terminateds = jnp.array([[False, False, True, False]])  # [1, 4] bool
    truncateds = jnp.array([[False, False, False, False]])  # [1, 4] bool
    next_values = jnp.array([[[10.0], [10.0], [10.0], [10.0]]], dtype=jnp.float32)  # [1, 4, 1]

    targets = jax.vmap(discount_with_terminated, in_axes=(0, 0, 0, 0, None))(
        rewards, terminateds, truncateds, next_values, GAMMA
    )

    assert targets.shape == (1, 4, 1)
    # Same episode-boundary returns as the [T,1] direct-call test, now via the
    # real flat+bool+vmap path: terminal at idx2 resets, forced truncation at the
    # last step bootstraps from next_value.
    assert [round(float(v), 2) for v in np.asarray(targets).reshape(-1)] == [2.71, 1.9, 1.0, 10.0]


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


@pytest.mark.parametrize("worker", [ApeXQWorker, ApeXDPGWorker, Impala_Worker])
def test_distributed_workers_store_termination_and_truncation_separately(worker):
    tree = ast.parse(textwrap.dedent(inspect.getsource(worker.run)))
    local_adds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "local_buffer"
        and node.func.attr == "add"
    ]

    assert len(local_adds) == 1
    assert ast.unparse(local_adds[0].args[-2]) == "terminated"
    assert ast.unparse(local_adds[0].args[-1]) == "truncated"


# --- A2C vectorized loop: dummy-step neutralization -----------------------
class _ScriptEnv:
    ws = 2

    def __init__(self, term_script):
        self._script = term_script
        self._t = 0
        self.sent_actions = []
        self.awaiting_result = False

    def current_obs(self):
        return np.zeros((self.ws, 2), dtype=np.float32)

    def step(self, actions):
        if self.awaiting_result:
            raise RuntimeError("step called while a result is pending")
        self.awaiting_result = True
        self.sent_actions.append(np.asarray(actions).copy())

    def get_result(self):
        if not self.awaiting_result:
            raise RuntimeError("get_result called without a pending step")
        self.awaiting_result = False
        terms = self._script[self._t]
        self._t += 1
        nxt = np.ones((self.ws, 2), dtype=np.float32)
        # Nonzero so the dummy-step reward zeroing is observable in the buffer.
        rewards = np.ones(self.ws, dtype=np.float32)
        truncs = np.zeros(self.ws, dtype=bool)
        return nxt, rewards, terms, truncs, {}


class _LivesScriptEnv(_ScriptEnv):
    def __init__(self, rows):
        self._rows = rows
        self._t = 0
        self.sent_actions = []
        self.awaiting_result = False

    def get_result(self):
        if not self.awaiting_result:
            raise RuntimeError("get_result called without a pending step")
        self.awaiting_result = False
        terms, truncs, lives = self._rows[self._t]
        self._t += 1
        return (
            np.ones((self.ws, 2), dtype=np.float32),
            np.ones(self.ws, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            {"lives": np.asarray(lives, dtype=np.int32)},
        )

    def real_reset_mask(self, terminateds, truncateds, infos):
        lives = np.asarray(infos["lives"], dtype=np.int32)
        return np.asarray(truncateds, dtype=bool) | (
            np.asarray(terminateds, dtype=bool) & (lives == 0)
        )

    def autoreset_mask(self, terminateds, truncateds, infos):
        return self.real_reset_mask(terminateds, truncateds, infos)


class _RecordingBuffer:
    def __init__(self):
        self.terminateds = []
        self.rewards = []
        self.truncateds = []
        self.actions = []

    def add(self, obs, action, reward, nxtobs, terminated, truncated):
        self.terminateds.append(np.asarray(terminated).copy())
        self.rewards.append(np.asarray(reward).copy())
        self.truncateds.append(np.asarray(truncated).copy())
        self.actions.append(np.asarray(action).copy())


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
    agent.action_type = "discrete"
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


def test_a2c_vectorized_converts_continuous_actions_before_env_step():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.env = _ScriptEnv([np.array([False, False])])
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 10_000
    agent.action_type = "continuous"
    agent.actions = lambda obs: np.full((2, 1), 6.0, dtype=np.float32)
    agent.conv_action = lambda actions: np.clip(actions, -3.0, 3.0) / 3.0
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([0]))

    assert len(agent.env.sent_actions) == 1
    assert all(np.array_equal(action, np.ones((2, 1))) for action in agent.env.sent_actions)
    assert np.array_equal(agent.buffer.actions[0], np.full((2, 1), 6.0))


def test_a2c_vectorized_recomputes_pipelined_action_after_policy_update():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.env = _ScriptEnv([np.array([False, False]), np.array([False, False])])
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 1
    agent.action_type = "discrete"
    agent.policy_version = 0
    agent.actions = lambda obs: np.full((2, 1), agent.policy_version, dtype=np.int32)

    train_calls = []

    def train_step(_steps, logger_run=None):
        train_calls.append((_steps, logger_run))
        agent.policy_version += 1
        return 0.0

    agent.train_step = train_step
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    ctx = _Ctx([0, 1])
    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, ctx)

    sent_versions = [int(actions[0, 0]) for actions in agent.env.sent_actions]
    assert sent_versions == [0, 1]
    assert train_calls == [(0, ctx.logger_run)]
    assert not hasattr(agent, "logger_run")


def test_a2c_vectorized_can_run_twice_without_a_pending_step():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.env = _ScriptEnv([np.array([False, False]), np.array([False, False])])
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 10_000
    agent.action_type = "discrete"
    agent.actions = lambda obs: np.zeros((2, 1), dtype=np.int32)
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([0]))
    assert not agent.env.awaiting_result

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([1]))
    assert not agent.env.awaiting_result


def test_a2c_vectorized_keeps_post_lifeloss_step_real():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.env = _LivesScriptEnv(
        [
            ([False, False], [False, False], [3, 3]),
            ([True, False], [False, False], [2, 3]),
            ([False, False], [False, False], [2, 3]),
        ]
    )
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 10_000
    agent.action_type = "discrete"
    agent.lossque = deque(maxlen=10)
    agent.actions = lambda obs: np.zeros((2, 1), dtype=np.float32)
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([0, 1, 2]))

    assert agent.buffer.terminateds[2].tolist() == [False, False]
    assert agent.buffer.rewards[2].tolist() == [1.0, 1.0]


def test_a2c_vectorized_drops_gymnasium_lifeloss_autoreset_dummy():
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    env = _LivesScriptEnv(
        [
            ([False, False], [False, False], [3, 3]),
            ([True, False], [False, False], [2, 3]),
            ([False, False], [False, False], [2, 3]),
        ]
    )

    def autoreset_mask(terminateds, truncateds, infos):
        return np.logical_or(terminateds, truncateds)

    env.autoreset_mask = autoreset_mask
    agent.env = env
    agent.buffer = _RecordingBuffer()
    agent.worker_size = 2
    agent.batch_size = 10_000
    agent.action_type = "discrete"
    agent.lossque = deque(maxlen=10)
    agent.actions = lambda obs: np.zeros((2, 1), dtype=np.float32)
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"

    Actor_Critic_Policy_Gradient_Family.learn_VectorizedEnv(agent, _Ctx([0, 1, 2]))

    assert agent.buffer.terminateds[2].tolist() == [True, False]
    assert agent.buffer.rewards[2].tolist() == [0.0, 1.0]


class _ScriptSingleEnv:
    """Minimal single (non-vectorized) gym-style env for the SingleEnv loop."""

    def __init__(self):
        self.received_actions = []

    def reset(self):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.received_actions.append(action)
        return np.ones(2, dtype=np.float32), 1.0, False, False, {}


def test_a2c_single_env_action_plumbing_and_buffer_shape():
    # Regression for the worker=1 SingleEnv path: it used to double-collapse the
    # action (`self.actions(obs)[0]` then `conv_action(...)[0]`), so conv_action's
    # own a[0] already left a scalar and the extra [0] raised IndexError before
    # env.step ever ran. The loop now mirrors eval: conv_action(self.actions(obs))
    # normalized exactly once.
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    env = _ScriptSingleEnv()
    agent.env = env
    agent.batch_size = 10_000  # train cadence never fires
    agent.lossque = deque(maxlen=10)
    # action_discrete returns [worker=1, action_dim=1]; _discrete_action_conv is a[0].
    agent.actions = lambda obs: np.array([[1]], dtype=np.int32)
    agent.conv_action = lambda a: a[0]
    agent.train_step = lambda steps: 0.0
    agent.eval = lambda ctx, steps: None
    agent.description = lambda eval_result: "desc"
    agent.buffer = _RecordingBuffer()

    Actor_Critic_Policy_Gradient_Family.learn_SingleEnv(agent, _Ctx([0, 1, 2]))

    # env.step received a bare scalar action each step (not an array, no crash).
    assert len(env.received_actions) == 3
    assert all(np.ndim(a) == 0 for a in env.received_actions)
    # The buffer keeps the worker dim so a real EpochBuffer's action[worker_idx]
    # stays indexable -- stored shape [worker=1, action_dim=1], matching the
    # vectorized path rather than the pre-fix collapsed scalar.
    assert all(a.shape == (1, 1) for a in agent.buffer.actions)


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
        agent.action_type = "discrete"
        agent.lossque = deque(maxlen=10)
        trained = []
        agent.actions = lambda obs: np.zeros(worker, dtype=np.int32)
        agent.train_step = lambda steps, logger_run=None: (trained.append(steps), 0.0)[1]
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
