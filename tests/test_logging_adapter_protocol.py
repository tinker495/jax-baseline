"""Logging Adapter Protocol seam tests (issue #14, ADR 0004).

Locks the unified ``LoggerRun`` contract, the removal of the TensorBoard-specific
leaks (raw writer, tuple-unpack, direct ``add_scalar`` / ``add_custom_scalars``),
the ``--logger`` selector, and the distributed Ray logger path going through the
protocol with the multiline layout degrading to a no-op for non-TensorBoard
backends.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from jax_baselines.core import runtime_adapters as ra
from jax_baselines.core.runtime_adapters import LoggerRun, NoOpLoggerRun

REPO = Path(__file__).resolve().parent.parent
PROTOCOL_METHODS = (
    "log_param",
    "log_metric",
    "log_histogram",
    "declare_multiline_layout",
    "get_local_path",
)


class InMemoryRun:
    """A LoggerRun test double — proves the seam end-to-end without TensorBoard."""

    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        self.metrics: list = []
        self.histograms: list = []
        self.params: list = []
        self.multilines: list = []

    def log_param(self, hparam_dict):
        self.params.append(dict(hparam_dict))

    def log_metric(self, key, value, step=None):
        self.metrics.append((key, value, step))

    def log_histogram(self, key, value, step=None):
        self.histograms.append((key, value, step))

    def declare_multiline_layout(self, eps):
        self.multilines.append(list(eps))

    def get_local_path(self, path):
        return os.path.join(self.local_dir, path)


class InMemoryLogger:
    """A LoggerFactory/context-manager test double returning one shared run."""

    def __init__(self, run_name, experiment_name, local_dir, agent):
        self.run = InMemoryRun(os.path.join(local_dir, experiment_name, run_name))
        self.hparams: list = []
        self.closed = False

    def log_hparams(self, agent_or_hparams):
        self.hparams.append(agent_or_hparams)

    def __enter__(self):
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def close(self):
        self.closed = True


# --- unified protocol surface -------------------------------------------------


def test_logger_run_protocol_has_unified_surface():
    for method in PROTOCOL_METHODS:
        assert hasattr(LoggerRun, method), method


def test_separate_histogram_protocol_is_folded_in():
    # log_histogram is now part of LoggerRun; the separate optional protocol is gone.
    assert not hasattr(ra, "HistogramLoggerRun")


def test_noop_logger_run_satisfies_protocol_runtime_check():
    run = NoOpLoggerRun("/tmp/run")
    assert isinstance(run, LoggerRun)
    assert run.log_metric("eval/episode_reward", 1.0, 3) is None
    assert run.log_histogram("loss/tau", [0.1, 0.9], 3) is None
    assert run.declare_multiline_layout([0.1, 0.4]) is None
    assert run.get_local_path("params") == os.path.join("/tmp/run", "params")


# --- leak removal -------------------------------------------------------------


def test_noop_run_exposes_no_raw_writer_or_tuple_unpack():
    run = NoOpLoggerRun("/tmp/run")
    assert not hasattr(run, "writer")
    assert not hasattr(type(run), "__iter__")
    assert not hasattr(ra, "NoOpWriter")


def test_core_logs_only_through_the_protocol():
    """AC#3: the Algorithm Core never accesses a raw TensorBoard writer, unpacks
    the logger as a tuple, or calls add_scalar / add_custom_scalars directly."""
    forbidden = (
        ".add_scalar",
        ".add_histogram",
        ".add_custom_scalars",
        ".add_summary",
        "as (summary",
        "NoOpWriter",
    )
    offenders = []
    for path in (REPO / "jax_baselines").rglob("*.py"):
        text = path.read_text()
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO)}: {pattern!r}")
    assert not offenders, "raw TensorBoard writer access leaked into core:\n" + "\n".join(offenders)


def test_tensorboard_run_has_no_public_writer_or_context_wrapper(tmp_path):
    from experiments import runtime_adapters as era

    assert not hasattr(era, "TensorboardContext")
    logger = era.TensorboardLogger("run", "exp", str(tmp_path), None)
    with logger as run:
        assert isinstance(run, LoggerRun)
        # the raw SummaryWriter is private; call sites see only the protocol.
        assert not hasattr(run, "writer")
        for method in PROTOCOL_METHODS:
            assert callable(getattr(run, method)), method


# --- --logger selector --------------------------------------------------------


def test_logger_selector_default_resolves_to_tensorboard(monkeypatch):
    monkeypatch.delenv("JAXBL_LOGGER", raising=False)  # default is env-driven now
    from experiments.cli import _loggers
    from experiments.runtime_adapters import TensorboardLogger

    parser = argparse.ArgumentParser()
    _loggers.add_logger_args(parser)
    args = parser.parse_args([])
    assert args.logger == "tensorboard"
    assert _loggers.resolve_logger_factory(args) is TensorboardLogger


def test_logger_selector_exposes_three_backends():
    from experiments.cli import _loggers

    assert _loggers.LOGGER_BACKENDS == ("tensorboard", "wandb", "aim")
    parser = argparse.ArgumentParser()
    _loggers.add_logger_args(parser)
    # the three backends are selectable; backend-specific config flags exist.
    for backend in ("tensorboard", "wandb", "aim"):
        assert parser.parse_args(["--logger", backend]).logger == backend
    assert parser.parse_args(["--aim_repo", "r"]).aim_repo == "r"


# --- distributed Ray logger path goes through the protocol --------------------


def test_distributed_logger_server_logs_through_protocol_without_tensorboard(tmp_path):
    """The centralized Ray logger actor maps to ONE run and logs only through the
    protocol; the multiline layout is a no-op-able protocol call, not a raw
    add_custom_scalars."""
    from jax_baselines.APE_X.common_servers import Logger_server

    created: dict = {}

    class _Logger(InMemoryLogger):
        def __init__(self, *args):
            super().__init__(*args)
            created["logger"] = self

    # Instantiate the actor class directly (no Ray) — the methods are plain.
    server = Logger_server(str(tmp_path), "run", logger_factory=_Logger)
    server.register_hparams({"learning_rate": 0.1})
    server.add_multiline([0.1, 0.4])
    server.log_trainer(5, {"loss/qloss": 2.0})
    server.log_worker({"rollout/episode_reward": 10.0}, episode=1)
    server.log_trainer(6, {"loss/qloss": 3.0})  # step advances -> worker buffer flushes
    server.last_update()

    logger = created["logger"]
    run = logger.run
    assert logger.hparams == [{"learning_rate": 0.1}]
    assert run.multilines == [[0.1, 0.4]]
    assert ("loss/qloss", 2.0, 5) in run.metrics
    assert ("loss/qloss", 3.0, 6) in run.metrics
    assert ("rollout/episode_reward", 10.0, 6) in run.metrics
    # get_log_dir returns the run's base directory (no trailing separator).
    assert server.get_log_dir() == os.path.normpath(run.get_local_path(""))


def test_multiline_layout_is_a_noop_on_non_tensorboard_backends():
    # NoOp degrades cleanly; an in-memory backend records intent but no raw layout.
    assert NoOpLoggerRun("/tmp/x").declare_multiline_layout([0.1, 0.4]) is None
    run = InMemoryRun("/tmp/x")
    run.declare_multiline_layout([0.1, 0.4])
    assert run.multilines == [[0.1, 0.4]]


def test_distributed_logger_server_close_finalizes_the_run(tmp_path):
    """The long-lived Ray logger actor finalizes its backend run explicitly via
    close(), not only on actor GC."""
    from jax_baselines.APE_X.common_servers import Logger_server

    created: dict = {}

    class _Logger(InMemoryLogger):
        def __init__(self, *args):
            super().__init__(*args)
            created["logger"] = self

    server = Logger_server(str(tmp_path), "run", logger_factory=_Logger)
    server.log_trainer(5, {"loss/qloss": 2.0})
    assert created["logger"].closed is False
    server.close()
    assert created["logger"].closed is True


def test_default_loggers_support_close(tmp_path):
    # The lifecycle close() the distributed actor calls exists on every logger.
    from experiments.runtime_adapters import TensorboardLogger
    from jax_baselines.core.runtime_adapters import NoOpLogger

    NoOpLogger("run", "exp", str(tmp_path), None).close()  # no-op, no crash
    tb = TensorboardLogger("run", "exp", str(tmp_path), None)
    tb.close()
    tb.close()  # idempotent


def test_noop_logger_supports_default_local_dir():
    from jax_baselines.core.runtime_adapters import NoOpLogger

    logger = NoOpLogger("run", "exp", None, None)

    assert logger.run.get_local_path("params") == os.path.join(".", "params")
