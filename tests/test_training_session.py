"""Tests for ``jax_baselines.core.training_session``.

The session owns the ``learn()`` lifecycle one seam above the rollout loop. The
win pinned here: a fake agent satisfying the *session contract* drives
``TrainingSession.run`` end-to-end with **no** env / buffer / model — retiring
the ``Q_Network_Family.__new__(...)`` instantiation hack the rollout suite needs.

Behaviors pinned:
- the lifecycle ordering ``run_name_update`` -> ``prepare_run`` ->
  ``run_training_loop`` -> ``eval`` -> ``save_params``;
- ``eval_freq`` is computed once and aligned down to ``worker_size``, and the
  agent sees exactly that value on ``ctx``;
- ``save_params`` is called once with the logger's local path.
"""

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.core.training_session import RunContext, TrainingSession


class FakeLoggerRun:
    def get_local_path(self, name):
        return ("local_path", name)


class FakeLogger:
    """Context-manager stand-in injected via ``logger_factory``."""

    def __init__(self, run_name, experiment_name, log_dir, agent):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.agent = agent
        self.run = FakeLoggerRun()

    def __enter__(self):
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeSessionAgent:
    """Minimal agent satisfying the session contract; no env/buffer/model."""

    def __init__(self, rec, worker_size=8, env_type="SingleEnv", use_checkpointing=False):
        self.rec = rec
        self.worker_size = worker_size
        self.env_type = env_type
        self.use_checkpointing = use_checkpointing
        self.log_dir = "/tmp/logs"
        self.log_interval = None
        self.seen_eval_freq = None

    def run_name_update(self, run_name):
        self.rec.append(("run_name_update", run_name))
        return run_name

    def prepare_run(self, total):
        self.rec.append(("prepare_run", total))

    def run_training_loop(self, ctx):
        self.rec.append(("run_training_loop",))
        self.seen_eval_freq = ctx.eval_freq

    def eval(self, ctx, steps):
        self.rec.append(("eval", steps))

    def save_params(self, path):
        self.rec.append(("save_params", path))


def _run(agent, total_timesteps=10000, log_interval=1000, eval_num=100):
    return TrainingSession().run(
        agent,
        total_timesteps,
        callback=None,
        log_interval=log_interval,
        experiment_name="exp",
        run_name="run",
        eval_num=eval_num,
        logger_factory=FakeLogger,
    )


def test_session_lifecycle_ordering():
    rec = []
    agent = FakeSessionAgent(rec)

    _run(agent)

    order = [e[0] for e in rec]
    assert order == [
        "run_name_update",
        "prepare_run",
        "run_training_loop",
        "eval",
        "save_params",
    ]


def test_eval_freq_aligned_to_worker_size():
    rec = []
    agent = FakeSessionAgent(rec, worker_size=8)

    _run(agent, total_timesteps=10000)

    # default eval_num=100: ((10000 // 100) // 8) * 8 == 96
    assert agent.seen_eval_freq == 96


def test_eval_num_controls_eval_freq():
    rec = []
    agent = FakeSessionAgent(rec, worker_size=8)

    _run(agent, total_timesteps=10000, eval_num=20)

    # 20 evals over the run: (10000 // 20) aligned down to worker_size 8 == 496
    assert agent.seen_eval_freq == 496


def test_eval_num_floored_at_one_eval():
    rec = []
    agent = FakeSessionAgent(rec, worker_size=8)

    # eval_num=0 must not divide-by-zero; floored to a single eval over the run.
    _run(agent, total_timesteps=10000, eval_num=0)

    assert agent.seen_eval_freq == 10000


def test_adapter_factories_are_injected_into_session():
    rec = []
    agent = FakeSessionAgent(rec)
    progress_calls = []
    record_test_fn = object()

    class FakeProgress:
        def __iter__(self):
            return iter(())

        def set_description(self, desc):
            rec.append(("set_description", desc))

    def progress_factory(*args, **kwargs):
        progress_calls.append((args, kwargs))
        return FakeProgress()

    TrainingSession().run(
        agent,
        total_timesteps=10000,
        callback=None,
        log_interval=1000,
        experiment_name="exp",
        run_name="run",
        eval_num=100,
        logger_factory=FakeLogger,
        progress_factory=progress_factory,
        record_test_fn=record_test_fn,
    )

    assert progress_calls == [((0, 10000, 8), {"miniters": 1000})]
    assert agent.record_test_fn is record_test_fn


def test_record_test_adapter_is_not_sticky_between_runs():
    rec = []
    agent = FakeSessionAgent(rec)

    def progress_factory(*args, **kwargs):
        return ()

    record_test_fn = object()

    TrainingSession().run(
        agent,
        total_timesteps=10000,
        callback=None,
        log_interval=1000,
        experiment_name="exp",
        run_name="run",
        eval_num=100,
        logger_factory=FakeLogger,
        progress_factory=progress_factory,
        record_test_fn=record_test_fn,
    )
    assert agent.record_test_fn is record_test_fn

    TrainingSession().run(
        agent,
        total_timesteps=10000,
        callback=None,
        log_interval=1000,
        experiment_name="exp",
        run_name="run",
        eval_num=100,
        logger_factory=FakeLogger,
        progress_factory=progress_factory,
        record_test_fn=None,
    )

    assert not hasattr(agent, "record_test_fn")


def test_save_params_called_once_with_logger_path():
    rec = []
    agent = FakeSessionAgent(rec)

    _run(agent)

    save_events = [e for e in rec if e[0] == "save_params"]
    assert save_events == [("save_params", ("local_path", "params"))]


def test_run_executes_without_env_buffer_or_model():
    """The whole ``run()`` drives a fake agent with no env/buffer/model."""
    rec = []
    agent = FakeSessionAgent(rec)

    _run(agent)

    assert agent.log_interval == 1000
    assert ("run_training_loop",) in rec
    assert isinstance(RunContext("lr", 96, "pbar", 1000), RunContext)


class FakeOnPolicySession(TrainingSession):
    def run(
        self,
        agent,
        total_timesteps,
        callback,
        log_interval,
        experiment_name,
        run_name,
        eval_num=100,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    ):
        agent.session_args = (
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
            eval_num,
            logger_factory,
            progress_factory,
            record_test_fn,
        )
        return "session-result"


class FakeOnPolicyAgent(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env_type="SingleEnv"):
        self.env_type = env_type
        self.calls = []
        self.eval_env = "eval-env"
        self.eval_eps = 3
        self.actions = "actions"
        self.conv_action = "conv-action"
        # Mirror the real base __init__ defaults (base_class.py:68,70) so the
        # hardened prepare_run (direct attribute access) keeps its no-op
        # semantics: lr_annealing off and params unset.
        self.lr_annealing = False
        self.params = None

    def learn_SingleEnv(self, ctx):
        self.calls.append(("single", ctx))

    def learn_VectorizedEnv(self, ctx):
        self.calls.append(("vector", ctx))


def test_on_policy_learn_delegates_to_training_session(monkeypatch):
    monkeypatch.setattr(
        "jax_baselines.A2C.base_class.TrainingSession",
        FakeOnPolicySession,
    )
    agent = FakeOnPolicyAgent()
    callback = object()

    result = agent.learn(
        1234,
        callback=callback,
        log_interval=7,
        experiment_name="exp",
        run_name="run",
    )

    assert result == "session-result"
    assert agent.session_args == (1234, callback, 7, "exp", "run", 100, None, None, None)


def test_on_policy_session_contract_hooks_are_minimal():
    agent = FakeOnPolicyAgent()

    assert agent.run_name_update("run") == "run"
    assert agent.prepare_run(100) is None
    assert not hasattr(agent, "update_eps")


def test_on_policy_run_training_loop_dispatches_single_env():
    agent = FakeOnPolicyAgent(env_type="SingleEnv")
    ctx = RunContext("logger-run", 10, "pbar", 5)

    agent.run_training_loop(ctx)

    assert agent.calls == [("single", ctx)]


def test_on_policy_run_training_loop_dispatches_vectorized_env():
    agent = FakeOnPolicyAgent(env_type="VectorizedEnv")
    ctx = RunContext("logger-run", 10, "pbar", 5)

    agent.run_training_loop(ctx)

    assert agent.calls == [("vector", ctx)]


def test_on_policy_eval_uses_run_context_logger(monkeypatch):
    rec = []

    def fake_evaluate_policy(eval_env, eval_eps, actions, logger_run, steps, conv_action):
        rec.append((eval_env, eval_eps, actions, logger_run, steps, conv_action))
        return {"score": 1.0}

    monkeypatch.setattr(
        "jax_baselines.A2C.base_class.evaluate_policy",
        fake_evaluate_policy,
    )
    agent = FakeOnPolicyAgent()
    ctx = RunContext("ctx-logger", 10, "pbar", 5)

    result = agent.eval(ctx, 42)

    assert result == {"score": 1.0}
    assert rec == [("eval-env", 3, "actions", "ctx-logger", 42, "conv-action")]
