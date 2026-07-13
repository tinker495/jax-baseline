"""Tests for the unified ``jax_baselines.core.rollout`` engine.

The bulk of this module is a characterization suite: golden call-traces pin the
observable behavior of the four ``learn_*`` loops for both the Q-Net and DPG
families. The traces were captured from the pre-refactor ``*RolloutLifecycle``
collaborators and must be reproduced byte-for-byte by ``RolloutEngine``. The
``_qnet_runner`` / ``_dpg_runner`` seam rebuilds the family ``RolloutSpec`` that
the base classes inject in production.

Behaviors deliberately pinned here:
- ordering reset -> actions -> env.step -> buffer.add -> (episode end) ->
  train-cadence -> eval -> description within each step;
- vectorized loops fire ``train_step`` *between* ``env.step`` and
  ``buffer.add`` (training sees pre-step buffer state);
- Q-Net refreshes epsilon inside the train-cadence branch for single-env loops
  (the next step observes it) but at the top of the loop every step for
  vectorized loops; DPG never refreshes epsilon;
- checkpointing loops skip the cadence ``train_step`` and train only via the
  checkpoint pulse callback;
- vectorized checkpointing defers pulse training until after the next
  ``env.step`` only when true-reset semantics are not needed;
- Q-Net single/vectorized checkpointing honor ``true_reset`` on a failed
  checkpoint, DPG does not.

The remaining tests cover :class:`CheckpointTrainPulse` and the base-class
action-selection seam directly.
"""

import numpy as np
import pytest

from jax_baselines.core.checkpoint import CheckpointController
from jax_baselines.core.eval import evaluate_policy, run_test_episodes
from jax_baselines.core.rollout import (
    ActionSelection,
    CheckpointTrainPulse,
    RolloutEngine,
    RolloutSpec,
)
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7


# --- runner seam: rebuilds the family RolloutSpec the base classes inject ---
def _spec(agent, single_action, vector_action, refresh_exploration, force_reset):
    return RolloutSpec(
        env=agent.env,
        replay_buffer=agent.replay_buffer,
        learning_starts=agent.learning_starts,
        train_freq=agent.train_freq,
        gradient_steps=agent.gradient_steps,
        eval_freq=agent.eval_freq,
        worker_size=agent.worker_size,
        single_action=single_action,
        vector_action=vector_action,
        refresh_exploration=refresh_exploration,
        force_reset=force_reset,
        train=agent.train_step,
        evaluate=agent.eval,
        describe=agent.description,
        bind_loss_window=lambda window: setattr(agent, "lossque", window),
        record_rollout_episode=lambda *a, **k: None,
        checkpoint_on_episode_end=agent._checkpoint_on_episode_end,
        checkpoint_pulse=agent.checkpointing_adapter.train_and_reset,
    )


def _qnet_runner(agent):
    def single_action(obs, steps):
        a = agent.actions(obs, agent.update_eps)
        return ActionSelection(a[0][0], a[0])

    def vector_action(obs, steps):
        a = agent.actions([obs], agent.update_eps)
        return ActionSelection(a, a)

    def refresh(steps):
        agent.update_eps = agent.exploration.value(steps)

    force_reset = agent.env.true_reset if agent._has_true_reset() else None
    return RolloutEngine(_spec(agent, single_action, vector_action, refresh, force_reset))


def _dpg_runner(agent):
    def single_action(obs, steps):
        a = agent.actions(obs, steps)
        return ActionSelection(a[0], a[0])

    def vector_action(obs, steps):
        a = agent.actions([obs], steps)
        return ActionSelection(a, a)

    return RolloutEngine(_spec(agent, single_action, vector_action, lambda steps: None, None))


# --- recording fakes -----------------------------------------------------
def rep(x):
    """Stable, numpy-free representation for trace events."""
    if isinstance(x, np.ndarray):
        return ("arr", rep(x.tolist()))
    if isinstance(x, list):
        return [rep(v) for v in x]
    if isinstance(x, tuple):
        return tuple(rep(v) for v in x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


class FakeSingleEnv:
    ws = 1

    def __init__(self, rec, script):
        self.rec = rec
        self.script = script  # list of (reward, terminated, truncated)
        self.t = 0
        self.c = 0

    def reset(self):
        self.rec.append(("reset",))
        self.c += 1
        return np.array([float(self.c)]), {}

    def true_reset(self):
        self.rec.append(("true_reset",))
        self.c += 1
        return np.array([float(self.c)]), {}

    def step(self, action):
        self.rec.append(("env_step", rep(action)))
        r, term, trunc = self.script[self.t]
        self.t += 1
        self.c += 1
        return np.array([float(self.c)]), r, term, trunc, {}


class FakeVecEnv:
    def __init__(self, rec, script, worker_size):
        self.rec = rec
        self.script = script  # list of (rewards, terms, truncs)
        self.t = 0
        self.c = 0
        self.ws = worker_size
        self._last = None

    def current_obs(self):
        self.c += 1
        return np.array([float(self.c)] * self.ws)

    def step(self, actions):
        self.rec.append(("env_step", rep(actions)))
        self._last = self.script[self.t]
        self.t += 1

    def get_result(self):
        rewards, terms, truncs = self._last
        nxt = np.array([float(self.c) + 0.5] * self.ws)
        return nxt, rewards, terms, truncs, {}

    def true_reset(self):
        self.rec.append(("true_reset",))


class FakeBuffer:
    def __init__(self, rec):
        self.rec = rec

    def add(self, obs, act, rew, nxt, term, trunc, store_mask=None):
        event = ("buffer_add", rep(act), rep(rew), rep(term), rep(trunc))
        if store_mask is not None:
            event += (("store_mask", rep(store_mask)),)
        self.rec.append(event)


class FakeExploration:
    def value(self, steps):
        return round(0.9 - 0.01 * steps, 4)


class _Adapter:
    def __init__(self, rec):
        self.rec = rec

    def train_and_reset(self, step_val, accumulated):
        self.rec.append(("ckpt_train_and_reset", step_val, int(accumulated)))


class FakeAgent:
    def __init__(self, rec, env, kind, ckpt_success_script=None, has_true_reset=False):
        self.rec = rec
        self.env = env
        self.kind = kind  # "qnet" | "dpg"
        self.replay_buffer = FakeBuffer(rec)
        self.exploration = FakeExploration()
        self.update_eps = 1.0
        self.learning_starts = 1
        self.train_freq = 2
        self.eval_freq = 3
        self.gradient_steps = 1
        self.worker_size = env.ws
        self.lossque = None
        self._ckpt_success_script = list(ckpt_success_script or [])
        self._has_tr = has_true_reset
        self.checkpointing_adapter = _Adapter(rec)

    def actions(self, obs, eps_or_steps):
        self.rec.append(("actions", rep(obs), rep(eps_or_steps)))
        fill = 7 if self.kind == "qnet" else 0.5
        return np.array([[fill] * self.worker_size]) if self.worker_size > 1 else np.array([[fill]])

    def train_step(self, steps, gradient_steps):
        self.rec.append(("train_step", steps, gradient_steps))
        return 0.1

    def eval(self, steps):
        self.rec.append(("eval", steps))
        return {"score": 1.0}

    def description(self, eval_result):
        self.rec.append(("description",))
        return "desc"

    def _has_true_reset(self):
        return self._has_tr

    def _checkpoint_on_episode_end(
        self, steps, score, eplen, train_and_reset_callback=None, advance_criterion=True
    ):
        self.rec.append(("checkpoint_on_episode_end", steps, rep(float(score)), int(eplen)))
        if not advance_criterion:
            return True
        if callable(train_and_reset_callback):
            train_and_reset_callback(steps, eplen)
        if self._ckpt_success_script:
            return self._ckpt_success_script.pop(0)
        return True


class FakePbar(list):
    def set_description(self, _):
        pass


SINGLE_SCRIPT = [
    (1.0, False, False),
    (1.0, True, False),
    (1.0, False, False),
    (1.0, False, True),
    (1.0, False, False),
    (1.0, False, False),
]


def _vec_script(ws):
    # Worker 0 terminates at step 1 (== learning_starts, so its score/eplen are
    # NOT reset here -- the reset only fires for steps > learning_starts) and then
    # truncates at step 3. The step-2 dummy autoreset is store_mask-excluded, so
    # the step-3 checkpoint reports the merged count (score=3, eplen=3) rather than
    # (4, 4). That cross-episode merge is a pre-existing warmup-boundary artifact,
    # unrelated to store_mask.
    def row(term_idx=None, trunc_idx=None):
        terms = np.zeros(ws, dtype=bool)
        truncs = np.zeros(ws, dtype=bool)
        if term_idx is not None:
            terms[term_idx] = True
        if trunc_idx is not None:
            truncs[trunc_idx] = True
        return np.ones(ws), terms, truncs

    return [row(), row(term_idx=0), row(), row(trunc_idx=0), row(), row()]


# --- scenarios -----------------------------------------------------------
def _run_qnet_single(rec):
    agent = FakeAgent(rec, FakeSingleEnv(rec, SINGLE_SCRIPT), "qnet")
    _qnet_runner(agent).learn_single_env(FakePbar(range(0, 6)))


def _run_qnet_vec(rec):
    agent = FakeAgent(rec, FakeVecEnv(rec, _vec_script(2), 2), "qnet")
    _qnet_runner(agent).learn_vectorized_env(FakePbar(range(0, 6)))


def _run_qnet_single_ckpt(rec):
    agent = FakeAgent(
        rec,
        FakeSingleEnv(rec, SINGLE_SCRIPT),
        "qnet",
        ckpt_success_script=[False],
        has_true_reset=True,
    )
    _qnet_runner(agent).learn_single_env_checkpointing(FakePbar(range(0, 6)))


def _run_qnet_vec_ckpt(rec):
    agent = FakeAgent(
        rec,
        FakeVecEnv(rec, _vec_script(2), 2),
        "qnet",
        ckpt_success_script=[False],
        has_true_reset=True,
    )
    _qnet_runner(agent).learn_vectorized_env_checkpointing(FakePbar(range(0, 6)))


def _run_dpg_single(rec):
    agent = FakeAgent(rec, FakeSingleEnv(rec, SINGLE_SCRIPT), "dpg")
    _dpg_runner(agent).learn_single_env(FakePbar(range(0, 6)))


def _run_dpg_vec(rec):
    agent = FakeAgent(rec, FakeVecEnv(rec, _vec_script(2), 2), "dpg")
    _dpg_runner(agent).learn_vectorized_env(FakePbar(range(0, 6)))


def _run_dpg_single_ckpt(rec):
    agent = FakeAgent(rec, FakeSingleEnv(rec, SINGLE_SCRIPT), "dpg")
    _dpg_runner(agent).learn_single_env_checkpointing(FakePbar(range(0, 6)))


def _run_dpg_vec_ckpt(rec):
    agent = FakeAgent(rec, FakeVecEnv(rec, _vec_script(2), 2), "dpg")
    _dpg_runner(agent).learn_vectorized_env_checkpointing(FakePbar(range(0, 6)))


SCENARIOS = {
    "qnet_single": _run_qnet_single,
    "qnet_vec": _run_qnet_vec,
    "qnet_single_ckpt": _run_qnet_single_ckpt,
    "qnet_vec_ckpt": _run_qnet_vec_ckpt,
    "dpg_single": _run_dpg_single,
    "dpg_vec": _run_dpg_vec,
    "dpg_single_ckpt": _run_dpg_single_ckpt,
    "dpg_vec_ckpt": _run_dpg_vec_ckpt,
}


GOLDEN = {
    "qnet_single": [
        ("reset",),
        ("actions", [("arr", [[1.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("eval", 0),
        ("actions", [("arr", [[2.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, True, False),
        ("reset",),
        ("actions", [("arr", [[4.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("train_step", 2, 1),
        ("actions", [("arr", [[5.0]])], 0.88),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, True),
        ("reset",),
        ("eval", 3),
        ("actions", [("arr", [[7.0]])], 0.88),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("train_step", 4, 1),
        ("actions", [("arr", [[8.0]])], 0.86),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
    ],
    "qnet_vec": [
        ("actions", [("arr", [1.0, 1.0])], 0.9),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
        ("eval", 0),
        ("actions", [("arr", [2.0, 2.0])], 0.89),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [True, False]),
            ("arr", [False, False]),
        ),
        ("actions", [("arr", [3.0, 3.0])], 0.88),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [4.0, 4.0])], 0.87),
        ("env_step", ("arr", [[7, 7]])),
        ("train_step", 3, 1),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [True, False]),
        ),
        ("eval", 3),
        ("actions", [("arr", [5.0, 5.0])], 0.86),
        ("env_step", ("arr", [[7, 7]])),
        ("train_step", 4, 1),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [6.0, 6.0])], 0.85),
        ("env_step", ("arr", [[7, 7]])),
        ("train_step", 5, 1),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
    ],
    "qnet_single_ckpt": [
        ("reset",),
        ("actions", [("arr", [[1.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("eval", 0),
        ("actions", [("arr", [[2.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, True, False),
        ("reset",),
        ("actions", [("arr", [[4.0]])], 1.0),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("actions", [("arr", [[5.0]])], 0.88),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, True),
        ("checkpoint_on_episode_end", 3, 2.0, 2),
        ("ckpt_train_and_reset", 3, 2),
        ("true_reset",),
        ("eval", 3),
        ("actions", [("arr", [[7.0]])], 0.88),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
        ("actions", [("arr", [[8.0]])], 0.86),
        ("env_step", 7),
        ("buffer_add", ("arr", [7]), 1.0, False, False),
    ],
    "qnet_vec_ckpt": [
        ("actions", [("arr", [1.0, 1.0])], 0.9),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
        ("eval", 0),
        ("actions", [("arr", [2.0, 2.0])], 0.89),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [True, False]),
            ("arr", [False, False]),
        ),
        ("actions", [("arr", [3.0, 3.0])], 0.88),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [4.0, 4.0])], 0.87),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [True, False]),
        ),
        ("checkpoint_on_episode_end", 3, 3.0, 3),
        ("ckpt_train_and_reset", 3, 3),
        ("true_reset",),
        ("eval", 3),
        ("actions", [("arr", [5.0, 5.0])], 0.86),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [6.0, 6.0])], 0.85),
        ("env_step", ("arr", [[7, 7]])),
        (
            "buffer_add",
            ("arr", [[7, 7]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
    ],
    "dpg_single": [
        ("reset",),
        ("actions", [("arr", [[1.0]])], 0),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("eval", 0),
        ("actions", [("arr", [[2.0]])], 1),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, True, False),
        ("reset",),
        ("actions", [("arr", [[4.0]])], 2),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("train_step", 2, 1),
        ("actions", [("arr", [[5.0]])], 3),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, True),
        ("reset",),
        ("eval", 3),
        ("actions", [("arr", [[7.0]])], 4),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("train_step", 4, 1),
        ("actions", [("arr", [[8.0]])], 5),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
    ],
    "dpg_vec": [
        ("actions", [("arr", [1.0, 1.0])], 0),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
        ("eval", 0),
        ("actions", [("arr", [2.0, 2.0])], 1),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [True, False]),
            ("arr", [False, False]),
        ),
        ("actions", [("arr", [3.0, 3.0])], 2),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [4.0, 4.0])], 3),
        ("env_step", ("arr", [[0.5, 0.5]])),
        ("train_step", 3, 1),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [True, False]),
        ),
        ("eval", 3),
        ("actions", [("arr", [5.0, 5.0])], 4),
        ("env_step", ("arr", [[0.5, 0.5]])),
        ("train_step", 4, 1),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [6.0, 6.0])], 5),
        ("env_step", ("arr", [[0.5, 0.5]])),
        ("train_step", 5, 1),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
    ],
    "dpg_single_ckpt": [
        ("reset",),
        ("actions", [("arr", [[1.0]])], 0),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("eval", 0),
        ("actions", [("arr", [[2.0]])], 1),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, True, False),
        ("reset",),
        ("actions", [("arr", [[4.0]])], 2),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("actions", [("arr", [[5.0]])], 3),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, True),
        ("checkpoint_on_episode_end", 3, 2.0, 2),
        ("ckpt_train_and_reset", 3, 2),
        ("reset",),
        ("eval", 3),
        ("actions", [("arr", [[7.0]])], 4),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
        ("actions", [("arr", [[8.0]])], 5),
        ("env_step", ("arr", [0.5])),
        ("buffer_add", ("arr", [0.5]), 1.0, False, False),
    ],
    "dpg_vec_ckpt": [
        ("actions", [("arr", [1.0, 1.0])], 0),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
        ("eval", 0),
        ("actions", [("arr", [2.0, 2.0])], 1),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [True, False]),
            ("arr", [False, False]),
        ),
        ("actions", [("arr", [3.0, 3.0])], 2),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [4.0, 4.0])], 3),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [True, False]),
        ),
        ("checkpoint_on_episode_end", 3, 3.0, 3),
        ("actions", [("arr", [5.0, 5.0])], 4),
        ("env_step", ("arr", [[0.5, 0.5]])),
        ("ckpt_train_and_reset", 3, 3),
        ("eval", 3),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
            ("store_mask", ("arr", [False, True])),
        ),
        ("actions", [("arr", [6.0, 6.0])], 5),
        ("env_step", ("arr", [[0.5, 0.5]])),
        (
            "buffer_add",
            ("arr", [[0.5, 0.5]]),
            ("arr", [1.0, 1.0]),
            ("arr", [False, False]),
            ("arr", [False, False]),
        ),
    ],
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_rollout_loop_call_trace_matches_golden(name):
    rec = []
    SCENARIOS[name](rec)
    assert rec == GOLDEN[name]


def test_vectorized_checkpointing_defers_pulse_until_after_next_env_step_without_true_reset():
    rec = []
    _run_dpg_vec_ckpt(rec)

    pulse_index = rec.index(("ckpt_train_and_reset", 3, 3))
    assert rec[pulse_index - 1] == ("env_step", ("arr", [[0.5, 0.5]]))
    assert rec[pulse_index + 1] == ("eval", 3)
    assert rec[pulse_index + 2][0] == "buffer_add"


def test_vectorized_checkpointing_keeps_pulse_immediate_when_true_reset_can_follow_failure():
    rec = []
    _run_qnet_vec_ckpt(rec)

    checkpoint_index = rec.index(("checkpoint_on_episode_end", 3, 3.0, 3))
    assert rec[checkpoint_index + 1] == ("ckpt_train_and_reset", 3, 3)
    assert rec[checkpoint_index + 2] == ("true_reset",)


def test_vectorized_checkpointing_flushes_deferred_pulse_after_normal_loop_exit():
    rec = []
    agent = FakeAgent(rec, FakeVecEnv(rec, _vec_script(2), 2), "dpg")

    _dpg_runner(agent).learn_vectorized_env_checkpointing(FakePbar(range(0, 4)))

    assert rec[-2:] == [("ckpt_train_and_reset", 3, 3), ("eval", 3)]


# --- CheckpointTrainPulse ------------------------------------------------
def _make_pulse(*, train_freq, gradient_steps, residual=0, post_pulse=None):
    state = {"residual": residual, "train_calls": [], "losses": []}

    def train(steps, gs):
        state["train_calls"].append((steps, gs))
        return 7.0

    pulse = CheckpointTrainPulse(
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        train=train,
        record_loss=state["losses"].append,
        read_residual=lambda: state["residual"],
        write_residual=lambda v: state.__setitem__("residual", v),
        post_pulse=post_pulse,
    )
    return pulse, state


def test_checkpoint_pulse_converts_timesteps_to_aligned_training_pulse():
    pulse, state = _make_pulse(train_freq=3, gradient_steps=2)

    pulse(50, 7)

    assert state["train_calls"] == [(50, 4)]  # 7 // 3 == 2 iters; 2 * 2 == 4 updates
    assert state["residual"] == 1  # 7 - 2 * 3
    assert state["losses"] == [7.0]


def test_checkpoint_pulse_carries_residual_and_skips_training_below_train_freq():
    pulse, state = _make_pulse(train_freq=5, gradient_steps=2, residual=1)

    pulse(10, 3)

    assert state["train_calls"] == []  # 1 + 3 == 4 < 5
    assert state["residual"] == 4
    assert state["losses"] == []


def test_checkpoint_pulse_runs_post_pulse_only_after_training():
    posts = []
    pulse, _ = _make_pulse(train_freq=3, gradient_steps=1, post_pulse=lambda: posts.append(True))

    pulse(1, 1)  # 1 < 3 -> no training, so policy-update snapshots must not move

    assert posts == []

    pulse(2, 2)

    assert posts == [True]


# --- base-class action-selection seam ------------------------------------
def test_qnet_single_action_double_indexes_discrete_action():
    agent = Q_Network_Family.__new__(Q_Network_Family)
    agent.update_eps = 0.5
    agent.actions = lambda obs, eps: np.array([[7]])

    sel = agent._single_action_selection(["obs"], steps=3)

    assert sel.env_action == 7
    assert list(sel.store_action) == [7]


def test_qnet_vector_action_passes_array_through():
    agent = Q_Network_Family.__new__(Q_Network_Family)
    agent.update_eps = 0.5
    arr = np.array([[7, 7]])
    agent.actions = lambda obs, eps: arr

    sel = agent._vector_action_selection("obs", steps=3)

    assert sel.env_action is arr
    assert sel.store_action is arr


def test_qnet_refresh_exploration_updates_epsilon():
    agent = Q_Network_Family.__new__(Q_Network_Family)
    agent.exploration = lambda steps: 0.42

    agent._refresh_exploration(steps=10)

    assert agent.update_eps == 0.42


def test_dpg_single_action_single_indexes_continuous_action():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    agent.actions = lambda obs, steps: np.array([[0.5]])

    sel = agent._single_action_selection(["obs"], steps=3)

    assert list(sel.env_action) == [0.5]
    assert list(sel.store_action) == [0.5]


def test_dpg_snapshot_action_normalizer_deepcopies_obs_rms_when_simba():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    agent.simba = True
    agent.obs_rms = {"mean": 1.0}

    agent._snapshot_action_normalizer()

    assert agent.action_obs_rms == {"mean": 1.0}
    assert agent.action_obs_rms is not agent.obs_rms


def test_dpg_snapshot_action_normalizer_noop_without_simba():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    agent.simba = False

    agent._snapshot_action_normalizer()

    assert not hasattr(agent, "action_obs_rms")


class _OffsetNormalizer:
    def __init__(self, offset):
        self.offset = offset
        self.calls = []

    def normalize(self, obs):
        self.calls.append(("normalize", obs))
        return [np.asarray(x) + self.offset for x in obs]

    def update(self, obs):
        self.calls.append(("update", obs))


class _ShapeCheckingEvalEnv:
    def __init__(self, action_shape):
        self.action_shape = action_shape
        self.actions = []

    def reset(self):
        return np.zeros(3), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_shape:
            raise ValueError(
                f"Action dimension mismatch. Expected {self.action_shape}, found {action.shape}"
            )
        self.actions.append(action)
        return np.zeros(3), 0.0, True, False, {}


def _dpg_action_agent():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    agent.simba = True
    agent.use_checkpointing = True
    agent.ckpt = type("Ckpt", (), {"enabled": True})()
    agent.obs_rms = _OffsetNormalizer(100)
    agent.action_obs_rms = _OffsetNormalizer(10)
    agent.checkpoint_obs_rms = _OffsetNormalizer(1000)
    agent.learning_starts = 0
    agent.worker_size = 1
    agent.action_size = (1,)
    agent._select_action_state = lambda eval, steps: {"encoder": None, "policy": None}
    agent._policy_action_from_state = lambda state, obs, eval, steps: np.asarray(obs[0])
    agent._apply_action_noise = lambda actions, steps, eval: actions
    return agent


def test_dpg_rollout_actions_use_policy_update_normalizer_and_update_live_rms():
    agent = _dpg_action_agent()

    action = agent.actions([np.array([1.0])], steps=5, eval=False)

    assert np.array_equal(action, np.array([11.0]))
    assert agent.obs_rms.calls == [("update", [np.array([1.0])])]
    assert agent.action_obs_rms.calls == [("normalize", [np.array([1.0])])]
    assert agent.checkpoint_obs_rms.calls == []


def test_dpg_eval_actions_use_checkpoint_normalizer():
    agent = _dpg_action_agent()

    action = agent.actions([np.array([1.0])], steps=5, eval=True)

    assert np.array_equal(action, np.array([1001.0]))
    assert agent.obs_rms.calls == []
    assert agent.action_obs_rms.calls == []
    assert agent.checkpoint_obs_rms.calls == [("normalize", [np.array([1.0])])]


def test_dpg_eval_skips_random_warmup_and_uses_policy_action():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    agent.simba = False
    agent.learning_starts = 100
    agent.worker_size = 32
    agent.action_size = (17,)
    agent._select_action_state = lambda eval, steps: {"encoder": None, "policy": None}
    agent._policy_action_from_state = lambda state, obs, eval, steps: np.full((1, 17), 0.5)
    agent._apply_action_noise = lambda actions, steps, eval: actions
    env = _ShapeCheckingEvalEnv((17,))

    evaluate_policy(env, 1, lambda obs: agent.actions(obs, steps=0, eval=True))

    assert env.actions[0].shape == (17,)
    assert np.all(env.actions[0] == 0.5)


def test_td3_test_action_uses_eval_action_shape_with_many_workers():
    agent = TD3.__new__(TD3)
    agent.simba = False
    agent.learning_starts = 0
    agent.worker_size = 32
    agent.action_size = (17,)
    agent.action_noise = 0.1
    agent._select_action_state = lambda eval, steps: {"encoder": None, "policy": None}
    agent._policy_action_from_state = lambda state, obs, eval, steps: np.ones((1, 17))
    env = _ShapeCheckingEvalEnv((17,))

    run_test_episodes(env, agent.test_action, episode=1)

    assert env.actions[0].shape == (17,)


def test_td7_eval_snapshot_waits_for_checkpoint_enabled_gate():
    agent = TD7.__new__(TD7)
    agent.use_checkpointing = True
    agent.ckpt = type("Ckpt", (), {"enabled": False})()
    agent.eval_snapshot = {"encoder": "checkpoint-encoder", "policy": "checkpoint-policy"}
    agent.fixed_encoder_params = "live-encoder"
    agent.policy_params = "live-policy"

    assert TD7._select_action_state(agent, eval=True, steps=5) == {
        "encoder": "live-encoder",
        "policy": "live-policy",
    }

    agent.ckpt.enabled = True

    assert TD7._select_action_state(agent, eval=True, steps=5) == agent.eval_snapshot


# --- engine + real CheckpointTrainPulse, end-to-end -----------------------
def test_checkpointing_loop_drives_real_pulse_to_train_step():
    """The checkpointing loop must drive the *real* pulse (not a stub): episode
    end -> checkpoint_on_episode_end -> CheckpointTrainPulse -> train_step with
    train_freq-aligned updates and a residual write-back on the agent."""
    rec = []
    env = FakeSingleEnv(rec, SINGLE_SCRIPT)
    agent = FakeAgent(rec, env, "qnet")
    agent._ckpt_update_residual = 0

    pulse = CheckpointTrainPulse(
        train_freq=agent.train_freq,  # 2
        gradient_steps=agent.gradient_steps,  # 1
        train=agent.train_step,
        record_loss=lambda loss: agent.lossque.append(loss),
        read_residual=lambda: agent._ckpt_update_residual,
        write_residual=lambda v: setattr(agent, "_ckpt_update_residual", v),
    )

    def single_action(obs, steps):
        a = agent.actions(obs, agent.update_eps)
        return ActionSelection(a[0][0], a[0])

    spec = RolloutSpec(
        env=env,
        replay_buffer=agent.replay_buffer,
        learning_starts=agent.learning_starts,
        train_freq=agent.train_freq,
        gradient_steps=agent.gradient_steps,
        eval_freq=agent.eval_freq,
        worker_size=1,
        single_action=single_action,
        vector_action=single_action,
        refresh_exploration=lambda steps: None,
        force_reset=None,
        train=agent.train_step,
        evaluate=agent.eval,
        describe=agent.description,
        bind_loss_window=lambda window: setattr(agent, "lossque", window),
        record_rollout_episode=lambda *a, **k: None,
        checkpoint_on_episode_end=agent._checkpoint_on_episode_end,
        checkpoint_pulse=pulse,
    )
    RolloutEngine(spec).learn_single_env_checkpointing(FakePbar(range(0, 6)))

    # Only episode end past learning_starts is step 3 (eplen=2). The real pulse
    # turns 2 accumulated timesteps into 2 // train_freq(2) = 1 update iter,
    # i.e. train(steps=3, gradient_steps=1*gradient_steps=1), residual -> 0.
    train_events = [e for e in rec if e[0] == "train_step"]
    assert train_events == [("train_step", 3, 1)]
    assert agent._ckpt_update_residual == 0
    assert list(agent.lossque) == [0.1]


def test_vectorized_checkpointing_pools_same_step_worker_lengths_before_pulse():
    rec = []
    worker_size = 2
    zeros = np.zeros(worker_size, dtype=bool)
    script = [
        (np.ones(worker_size), zeros.copy(), zeros.copy()),
        (np.ones(worker_size), zeros.copy(), zeros.copy()),
        (np.ones(worker_size), zeros.copy(), np.ones(worker_size, dtype=bool)),
    ]
    env = FakeVecEnv(rec, script, worker_size)
    agent = FakeAgent(rec, env, "qnet")
    pulse_calls = []
    controller = CheckpointController(
        use_checkpointing=True,
        steps_before_checkpointing=1000,
        max_eps_before_checkpointing=1,
        initial_window=1,
        baseline_q=0.2,
        baseline_mode="mean",
        use_return_standardization=False,
        snapshot=lambda: None,
    )

    def vector_action(obs, steps):
        actions = agent.actions([obs], agent.update_eps)
        return ActionSelection(actions, actions)

    spec = RolloutSpec(
        env=env,
        replay_buffer=agent.replay_buffer,
        learning_starts=agent.learning_starts,
        train_freq=agent.train_freq,
        gradient_steps=agent.gradient_steps,
        eval_freq=agent.eval_freq,
        worker_size=worker_size,
        single_action=vector_action,
        vector_action=vector_action,
        refresh_exploration=lambda steps: None,
        force_reset=None,
        train=agent.train_step,
        evaluate=agent.eval,
        describe=agent.description,
        bind_loss_window=lambda window: setattr(agent, "lossque", window),
        record_rollout_episode=lambda *args, **kwargs: None,
        checkpoint_on_episode_end=controller.on_episode_end,
        checkpoint_pulse=lambda steps, accumulated: pulse_calls.append((steps, accumulated)),
    )

    RolloutEngine(spec).learn_vectorized_env_checkpointing(FakePbar(range(0, 3)))

    assert pulse_calls == [(2, 6)]


# --- Atari episodic_life autoreset: only a real reset spawns a dummy ---------
class _LivesVecEnv:
    """Vec env that scripts per-step (terminated, truncated, lives) and emits
    ``lives`` in info, exercising the Atari episodic_life autoreset path."""

    def __init__(self, rec, steps):
        self.rec = rec
        self.steps = steps  # list of (terms, truncs, lives), each length ws
        self.ws = len(steps[0][0])
        self.t = 0
        self.c = 0
        self._last = None

    def current_obs(self):
        self.c += 1
        return np.array([float(self.c)] * self.ws)

    def step(self, actions):
        self.rec.append(("env_step", rep(actions)))
        self._last = self.steps[self.t]
        self.t += 1

    def get_result(self):
        terms, truncs, lives = self._last
        nxt = np.array([float(self.c) + 0.5] * self.ws)
        rewards = np.ones(self.ws)
        infos = {"lives": np.asarray(lives, dtype=np.int32)}
        return (
            nxt,
            rewards,
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    def real_reset_mask(self, terminateds, truncateds, infos):
        lives = np.asarray(infos["lives"], dtype=np.int32)
        return np.asarray(truncateds, dtype=bool) | (
            np.asarray(terminateds, dtype=bool) & (lives == 0)
        )

    def autoreset_mask(self, terminateds, truncateds, infos):
        return self.real_reset_mask(terminateds, truncateds, infos)


def _store_mask_of(buffer_add_event):
    if len(buffer_add_event) == 6 and buffer_add_event[-1][0] == "store_mask":
        return buffer_add_event[-1][1]
    return None


def test_vectorized_rollout_retains_post_lifeloss_step_drops_real_reset_dummy():
    # Under Atari episodic_life, envpool reports ``terminated`` on a LIFE LOSS
    # (lives>0) but keeps the same episode running -- the next step is a real,
    # action-applied transition and must NOT be store_masked. Only a truncation
    # or the lives==0 game-over spawns an autoreset dummy that IS dropped. Pins
    # the bug where ``store_mask = ~prev_done`` dropped the post-life-loss step.
    rec = []
    steps = [
        ([False, False], [False, False], [3, 3]),  # 0 normal
        ([True, False], [False, False], [2, 3]),  # 1 worker0 life-loss (lives>0)
        ([False, False], [False, False], [2, 3]),  # 2 worker0 real continuation
        ([True, False], [False, False], [0, 3]),  # 3 worker0 game-over (lives==0)
        ([False, False], [False, False], [3, 3]),  # 4 worker0 autoreset dummy
        ([False, False], [True, False], [2, 3]),  # 5 worker0 truncation (lives>0)
        ([False, False], [False, False], [3, 3]),  # 6 worker0 dummy after truncation
    ]
    agent = FakeAgent(rec, _LivesVecEnv(rec, steps), "qnet")
    _qnet_runner(agent).learn_vectorized_env(FakePbar(range(0, 7)))

    adds = [e for e in rec if e[0] == "buffer_add"]
    assert len(adds) == 7
    # step2: the real post-life-loss transition is kept (no worker masked out).
    assert _store_mask_of(adds[2]) is None
    # step4: the genuine lives==0 game-over autoreset dummy is dropped (worker0).
    assert _store_mask_of(adds[4]) == ("arr", [False, True])
    # step6: the truncation autoreset dummy is dropped (worker0).
    assert _store_mask_of(adds[6]) == ("arr", [False, True])


def test_vectorized_rollout_preserves_train_freq_replay_ratio():
    rec = []
    worker_size = 8
    script = [
        (
            np.ones(worker_size),
            np.zeros(worker_size, dtype=bool),
            np.zeros(worker_size, dtype=bool),
        )
        for _ in range(3)
    ]
    agent = FakeAgent(rec, FakeVecEnv(rec, script, worker_size), "qnet")
    agent.learning_starts = 0
    agent.train_freq = 4
    agent.gradient_steps = 1

    _qnet_runner(agent).learn_vectorized_env(FakePbar(range(8, 32, 8)))

    assert [event for event in rec if event[0] == "train_step"] == [
        ("train_step", 8, 2),
        ("train_step", 16, 2),
        ("train_step", 24, 2),
    ]
