"""W&B (#15) and Aim (#16) logging backend tests.

Both backends are optional extras that are NOT installed in the test env, so:

- selecting one uninstalled raises a clear, actionable ``SystemExit``;
- against a fake SDK module injected into ``sys.modules``, each backend satisfies
  the ``LoggerRun`` protocol and routes metrics / histograms / hparams / multiline
  correctly, mapping the re-entrant distributed logger server to exactly ONE run.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from jax_baselines.core.runtime_adapters import LoggerRun


def _args(**kw):
    defaults = dict(
        logger="tensorboard",
        wandb_entity=None,
        wandb_mode=None,
        aim_repo=None,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


# --- fake W&B SDK ------------------------------------------------------------


class _FakeHistogram:
    def __init__(self, value):
        self.value = value


class _FakeConfig:
    def __init__(self):
        self.data = {}

    def update(self, d, allow_val_change=False):
        self.data.update(d)


class _FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.config = _FakeConfig()
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((dict(data), step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.init_count = 0
        self.run = _FakeWandbRun()
        self.Histogram = _FakeHistogram

    def init(self, **kwargs):
        self.init_count += 1
        self.init_kwargs = kwargs
        return self.run


# --- fake Aim SDK ------------------------------------------------------------


class _FakeDistribution:
    def __init__(self, value):
        self.value = value


class _FakeAimRun:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.tracked = []
        self.params = {}
        self.name = None
        self.closed = False

    def track(self, value, name=None, step=None):
        self.tracked.append((value, name, step))

    def __setitem__(self, key, value):
        self.params[key] = value

    def close(self):
        self.closed = True


class _FakeAim:
    def __init__(self):
        self.last_run = None
        self.run_count = 0
        self.Distribution = _FakeDistribution

    def Run(self, **kwargs):
        self.run_count += 1
        self.last_run = _FakeAimRun(**kwargs)
        return self.last_run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def fake_aim(monkeypatch):
    fake = _FakeAim()
    monkeypatch.setitem(sys.modules, "aim", fake)
    return fake


# --- uninstalled -> clear, actionable error ----------------------------------


def test_wandb_uninstalled_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # force ImportError
    from experiments.loggers.wandb_logger import make_wandb_logger_factory

    with pytest.raises(SystemExit, match="wandb"):
        make_wandb_logger_factory(_args(logger="wandb"))


def test_aim_uninstalled_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)
    from experiments.loggers.aim_logger import make_aim_logger_factory

    with pytest.raises(SystemExit, match="aim"):
        make_aim_logger_factory(_args(logger="aim"))


# --- W&B backend against the protocol ----------------------------------------


def test_wandb_backend_satisfies_protocol_and_routes(fake_wandb, tmp_path):
    from experiments.loggers.wandb_logger import make_wandb_logger_factory

    factory = make_wandb_logger_factory(
        _args(logger="wandb", wandb_entity="team", wandb_mode="offline")
    )
    logger = factory("run", "exp", str(tmp_path), None)
    with logger as run:
        assert isinstance(run, LoggerRun)
        run.log_metric("eval/episode_reward", 1.0, 3)
        run.log_metric("loss/qloss", 2.0, 3)
        run.log_histogram("loss/tau", [0.1, 0.9], 3)
        run.log_param({"learning_rate": 0.1})
        assert run.declare_multiline_layout([0.1, 0.4]) is None  # no-op on W&B
        assert run.get_local_path("params").endswith("params")

    assert fake_wandb.init_kwargs["project"] == "exp"  # project = experiment_name
    assert fake_wandb.init_kwargs["entity"] == "team"
    assert fake_wandb.init_kwargs["name"] == "run"
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert ({"eval/episode_reward": 1.0}, 3) in fake_wandb.run.logged
    assert ({"loss/qloss": 2.0}, 3) in fake_wandb.run.logged
    assert fake_wandb.run.config.data["learning_rate"] == 0.1
    histograms = [
        data
        for data, _ in fake_wandb.run.logged
        if any(isinstance(v, _FakeHistogram) for v in data.values())
    ]
    assert len(histograms) == 1


def test_wandb_coerces_array_metric_to_python_float(fake_wandb, tmp_path):
    # The core emits JAX/NumPy scalars; the backend must hand W&B a plain float.
    from experiments.loggers.wandb_logger import make_wandb_logger_factory

    with make_wandb_logger_factory(_args(logger="wandb"))("run", "exp", str(tmp_path), None) as run:
        run.log_metric("loss/q", np.array(2.5), 1)
    value = fake_wandb.run.logged[0][0]["loss/q"]
    assert isinstance(value, float) and value == 2.5


def test_aim_coerces_array_metric_to_python_float(fake_aim, tmp_path):
    from experiments.loggers.aim_logger import make_aim_logger_factory

    with make_aim_logger_factory(_args(logger="aim"))("run", "exp", str(tmp_path), None) as run:
        run.log_metric("loss/q", np.array(2.5), 1)
    tracked_value = fake_aim.last_run.tracked[0][0]
    assert isinstance(tracked_value, float) and tracked_value == 2.5


def test_wandb_project_defaults_to_experiment_name(fake_wandb, tmp_path):
    from experiments.loggers.wandb_logger import make_wandb_logger_factory

    factory = make_wandb_logger_factory(_args(logger="wandb"))
    with factory("run", "my_experiment", str(tmp_path), None):
        pass
    assert fake_wandb.init_kwargs["project"] == "my_experiment"


def test_wandb_local_run_directories_are_unique(fake_wandb, tmp_path):
    from experiments.loggers.wandb_logger import make_wandb_logger_factory

    factory = make_wandb_logger_factory(_args(logger="wandb"))

    first = factory("run", "exp", str(tmp_path), None)
    second = factory("run", "exp", str(tmp_path), None)

    assert first._local_dir.endswith("exp/run_01")
    assert second._local_dir.endswith("exp/run_02")


def test_wandb_distributed_logger_server_maps_to_one_run(fake_wandb, tmp_path):
    """AC#15: a Ray distributed family logs end-to-end into a single W&B run."""
    from experiments.loggers.wandb_logger import make_wandb_logger_factory
    from jax_baselines.APE_X.common_servers import Logger_server

    factory = make_wandb_logger_factory(_args(logger="wandb"))
    server = Logger_server(str(tmp_path), "run", "dist_exp", factory)
    server.register_hparams({"learning_rate": 0.1})
    server.add_multiline([0.1, 0.4])
    server.log_trainer(5, {"loss/qloss": 2.0})
    server.log_worker({"rollout/episode_reward": 10.0}, episode=1)
    server.log_trainer(6, {"loss/qloss": 3.0})
    server.last_update()
    server.close()

    assert fake_wandb.init_count == 1  # one W&B run for the entire actor
    assert fake_wandb.init_kwargs["project"] == "dist_exp"  # experiment_name threaded in
    assert ({"loss/qloss": 2.0}, 5) in fake_wandb.run.logged
    assert ({"rollout/episode_reward": 10.0}, 6) in fake_wandb.run.logged
    assert fake_wandb.run.config.data == {"learning_rate": 0.1}
    assert fake_wandb.run.finished is True  # finalized explicitly, not via GC


# --- Aim backend against the protocol ----------------------------------------


def test_aim_backend_satisfies_protocol_and_routes(fake_aim, tmp_path):
    from experiments.loggers.aim_logger import make_aim_logger_factory

    factory = make_aim_logger_factory(_args(logger="aim", aim_repo="/tmp/aimrepo"))
    logger = factory("run", "exp", str(tmp_path), None)
    with logger as run:
        assert isinstance(run, LoggerRun)
        run.log_metric("eval/episode_reward", 1.0, 3)
        run.log_histogram("loss/tau", [0.1, 0.9], 3)
        run.log_param({"learning_rate": 0.1})
        assert run.declare_multiline_layout([0.1, 0.4]) is None  # no-op on Aim
        assert run.get_local_path("video").endswith("video")

    aim_run = fake_aim.last_run
    assert aim_run.init_kwargs["repo"] == "/tmp/aimrepo"
    assert aim_run.init_kwargs["experiment"] == "exp"
    assert aim_run.name == "run"  # run name reuses the run-name tagging
    assert (1.0, "eval/episode_reward", 3) in aim_run.tracked
    assert aim_run.params["hparams"] == {"learning_rate": 0.1}
    distributions = [v for v, _, _ in aim_run.tracked if isinstance(v, _FakeDistribution)]
    assert len(distributions) == 1


def test_aim_distributed_logger_server_maps_to_one_run(fake_aim, tmp_path):
    """AC#16: a Ray distributed family logs end-to-end into a single Aim Run."""
    from experiments.loggers.aim_logger import make_aim_logger_factory
    from jax_baselines.APE_X.common_servers import Logger_server

    factory = make_aim_logger_factory(_args(logger="aim"))
    server = Logger_server(str(tmp_path), "run", "dist_exp", factory)
    server.register_hparams({"learning_rate": 0.1})
    server.add_multiline([0.1, 0.4])
    server.log_trainer(5, {"loss/qloss": 2.0})
    server.last_update()
    server.close()

    assert fake_aim.run_count == 1  # one Aim Run for the entire actor
    assert fake_aim.last_run.init_kwargs["experiment"] == "dist_exp"  # experiment_name threaded in
    assert (2.0, "loss/qloss", 5) in fake_aim.last_run.tracked
    assert fake_aim.last_run.params["hparams"] == {"learning_rate": 0.1}
    assert fake_aim.last_run.closed is True  # finalized explicitly, not via GC


def test_aim_local_run_directories_are_unique(fake_aim, tmp_path):
    from experiments.loggers.aim_logger import make_aim_logger_factory

    factory = make_aim_logger_factory(_args(logger="aim"))

    first = factory("run", "exp", str(tmp_path), None)
    second = factory("run", "exp", str(tmp_path), None)

    assert first._local_dir.endswith("exp/run_01")
    assert second._local_dir.endswith("exp/run_02")


# --- --logger registry dispatch ----------------------------------------------


def test_logger_registry_dispatches_to_wandb(fake_wandb, tmp_path):
    from experiments.cli._loggers import resolve_logger_factory
    from experiments.loggers.wandb_logger import WandbLogger

    factory = resolve_logger_factory(_args(logger="wandb"))
    assert isinstance(factory("run", "exp", str(tmp_path), None), WandbLogger)


def test_logger_registry_dispatches_to_aim(fake_aim, tmp_path):
    from experiments.cli._loggers import resolve_logger_factory
    from experiments.loggers.aim_logger import AimLogger

    factory = resolve_logger_factory(_args(logger="aim"))
    assert isinstance(factory("run", "exp", str(tmp_path), None), AimLogger)
