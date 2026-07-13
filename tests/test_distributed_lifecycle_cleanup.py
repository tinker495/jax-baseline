from __future__ import annotations

from importlib import import_module
from threading import Event
from unittest.mock import MagicMock, Mock, call

import pytest

FAMILIES = (
    ("jax_baselines.APE_X.base_class", "Ape_X_Family", "apex"),
    (
        "jax_baselines.APE_X.dpg_base_class",
        "Ape_X_Deteministic_Policy_Gradient_Family",
        "apex_dpg",
    ),
    ("jax_baselines.IMPALA.base_class", "IMPALA_Family", "impala"),
)


def _make_agent(monkeypatch, module_name, class_name, kind, *, ready, stop_on_run):
    module = import_module(module_name)
    family = getattr(module, class_name)
    agent = family.__new__(family)
    stop = Event()
    runtime = Mock()
    runtime.create_event.side_effect = [stop, Event()]
    runtime.create_param_server.return_value = object()
    logger = runtime.create_logger_server.return_value
    logger.get_log_dir.return_value = "/tmp/run"

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.jax, "device_put", lambda params, _device: params)
    monkeypatch.setattr(module.jax, "devices", lambda _platform: [None])
    if kind == "apex":
        monkeypatch.setattr(module, "get_hyper_params", lambda _agent: {})

    agent.runtime = runtime
    worker = Mock()
    worker.run.side_effect = lambda *a, **kw: (stop.set() if stop_on_run else None) or "job"
    agent.workers = [worker]
    agent.params = {"weight": 1}
    agent.model_builder = object()
    agent.actor_builder = object()
    agent.worker_replay_factory = lambda *_args, **_kwargs: object()
    agent.seed = 1
    agent.log_dir = None
    agent.env_type = "SingleEnv"
    agent.n_step_method = False
    agent.n_step = 1

    if kind == "apex":
        agent.munchausen = False
        agent.param_noise = False
        agent.dueling_model = False
        agent.double_q = False
        agent.learning_starts = 1
        agent.exploration_initial_eps = 0.9
        agent.exploration_decay = 0.7
        agent.gradient_steps = 1
        agent.target_network_update_freq = 10
    elif kind == "apex_dpg":
        agent.learning_starts = 1
        agent.exploration_initial_eps = 0.9
        agent.exploration_decay = 0.7
        agent.gradient_steps = 1
        agent.param_broadcast_freq = 10
    else:
        agent.worker_num = 1
        agent.batch_size = 1
        agent.update_freq = 10

    if kind.startswith("apex"):
        agent.replay_buffer = MagicMock()
        agent.replay_buffer.__len__.return_value = int(ready)
        agent.replay_buffer.buffer_info.return_value = "buffer"
    else:
        agent.buffer = Mock()
        agent.buffer.queue_is_empty.return_value = not ready
        agent.buffer.queue_info.return_value = "queue"

    lifecycle = Mock()
    runtime.wait = lifecycle.wait
    agent.save_params = lifecycle.save_params
    logger.last_update = lifecycle.last_update
    logger.close = lifecycle.close
    runtime.shutdown = lifecycle.shutdown
    return agent, stop, lifecycle


@pytest.mark.parametrize("module_name,class_name,kind", FAMILIES)
def test_distributed_learn_cleans_up_after_warmup_stop(monkeypatch, module_name, class_name, kind):
    agent, stop, lifecycle = _make_agent(
        monkeypatch, module_name, class_name, kind, ready=False, stop_on_run=True
    )

    agent.learn(1, progress_factory=lambda *_args, **_kwargs: [])

    assert stop.is_set()
    assert lifecycle.mock_calls == [
        call.wait(["job"], timeout=300),
        call.save_params("/tmp/run"),
        call.last_update(),
        call.close(),
        call.shutdown(),
    ]


@pytest.mark.parametrize("module_name,class_name,kind", FAMILIES)
def test_distributed_learn_cleans_up_after_training_exception(
    monkeypatch, module_name, class_name, kind
):
    agent, stop, lifecycle = _make_agent(
        monkeypatch, module_name, class_name, kind, ready=True, stop_on_run=False
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("train failed")

    agent.train_step = fail

    with pytest.raises(RuntimeError, match="train failed"):
        agent.learn(1, progress_factory=lambda *_args, **_kwargs: [1])

    assert stop.is_set()
    assert lifecycle.mock_calls == [
        call.wait(["job"], timeout=300),
        call.last_update(),
        call.close(),
        call.shutdown(),
    ]


def test_ray_runtime_shutdown_is_idempotent(monkeypatch):
    from experiments import distributed_runtime

    calls = []

    class Manager:
        def shutdown(self):
            calls.append("manager.shutdown")

    class Ray:
        def shutdown(self):
            calls.append("ray.shutdown")

    runtime = distributed_runtime.RayDistributedRuntime.__new__(
        distributed_runtime.RayDistributedRuntime
    )
    runtime._manager = Manager()
    monkeypatch.setattr(distributed_runtime, "_ray", lambda: Ray())

    runtime.shutdown()
    runtime.shutdown()

    assert calls == ["manager.shutdown", "ray.shutdown"]


def test_ray_runtime_shuts_down_ray_when_manager_start_fails(monkeypatch):
    from experiments import distributed_runtime

    calls = []

    class Context:
        def Manager(self):
            raise RuntimeError("manager failed")

    class Ray:
        def init(self, **kwargs):
            calls.append(("ray.init", kwargs))

        def shutdown(self):
            calls.append(("ray.shutdown",))

    monkeypatch.setattr(distributed_runtime, "_ray", lambda: Ray())
    contexts = []
    monkeypatch.setattr(
        distributed_runtime.mp,
        "get_context",
        lambda name: contexts.append(name) or Context(),
    )

    with pytest.raises(RuntimeError, match="manager failed"):
        distributed_runtime.RayDistributedRuntime(num_cpus=3, num_gpus=0)

    assert calls == [
        (
            "ray.init",
            {"num_cpus": 3, "num_gpus": 0, "ignore_reinit_error": True},
        ),
        ("ray.shutdown",),
    ]
    assert contexts == ["spawn"]
