"""Tests for ``jax_baselines.common.training_session``.

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

from jax_baselines.common.training_session import RunContext, TrainingSession


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


def _run(agent, total_timesteps=10000, log_interval=1000):
    return TrainingSession().run(
        agent,
        total_timesteps,
        callback=None,
        log_interval=log_interval,
        experiment_name="exp",
        run_name="run",
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

    # ((10000 // 100) // 8) * 8 == 96
    assert agent.seen_eval_freq == 96


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
