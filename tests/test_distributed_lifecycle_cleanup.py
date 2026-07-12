from __future__ import annotations

from importlib import import_module

import pytest


class _Event:
    def __init__(self):
        self.value = False

    def clear(self):
        self.value = False

    def set(self):
        self.value = True

    def is_set(self):
        return self.value


class _Logger:
    def __init__(self, calls):
        self.calls = calls

    def register_hparams(self, _hparams):
        self.calls.append("logger.register_hparams")

    def add_multiline(self, _eps):
        self.calls.append("logger.add_multiline")

    def get_log_dir(self):
        self.calls.append("logger.get_log_dir")
        return "/tmp/run"

    def last_update(self):
        self.calls.append("logger.last_update")

    def close(self):
        self.calls.append("logger.close")


class _Runtime:
    def __init__(self):
        self.calls = []
        self.events = []
        self.logger = _Logger(self.calls)

    def create_event(self):
        event = _Event()
        self.events.append(event)
        return event

    def create_param_server(self, _params):
        return object()

    def create_logger_server(self, *_args, **_kwargs):
        self.calls.append("runtime.create_logger")
        return self.logger

    def wait(self, jobs, timeout=None):
        self.calls.append(("runtime.wait", tuple(jobs), timeout))

    def shutdown(self):
        self.calls.append("runtime.shutdown")


class _Worker:
    def __init__(self, runtime, stop_on_run):
        self.runtime = runtime
        self.stop_on_run = stop_on_run

    def run(self, *_args, **_kwargs):
        if self.stop_on_run:
            self.runtime.events[0].set()
        return "job"


class _ApexReplay:
    def __init__(self, ready):
        self.ready = ready

    def __len__(self):
        return int(self.ready)

    def buffer_info(self):
        return "buffer"


class _ImpalaBuffer:
    def __init__(self, ready, calls):
        self.ready = ready
        self.calls = calls

    def queue_is_empty(self):
        return not self.ready

    def queue_info(self):
        return "queue"

    def clear(self):
        self.calls.append("buffer.clear")


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
    runtime = _Runtime()

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.jax, "device_put", lambda params, _device: params)
    monkeypatch.setattr(module.jax, "devices", lambda _platform: [None])
    if kind == "apex":
        monkeypatch.setattr(module, "get_hyper_params", lambda _agent: {})

    agent.runtime = runtime
    agent.workers = [_Worker(runtime, stop_on_run)]
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
        agent.replay_buffer = _ApexReplay(ready)
    elif kind == "apex_dpg":
        agent.learning_starts = 1
        agent.exploration_initial_eps = 0.9
        agent.exploration_decay = 0.7
        agent.gradient_steps = 1
        agent.param_broadcast_freq = 10
        agent.replay_buffer = _ApexReplay(ready)
    else:
        agent.worker_num = 1
        agent.batch_size = 1
        agent.update_freq = 10
        agent.buffer = _ImpalaBuffer(ready, runtime.calls)

    agent.save_params = lambda path: runtime.calls.append(("save_params", path))
    return agent, runtime


@pytest.mark.parametrize("module_name,class_name,kind", FAMILIES)
def test_distributed_learn_cleans_up_after_warmup_stop(monkeypatch, module_name, class_name, kind):
    agent, runtime = _make_agent(
        monkeypatch, module_name, class_name, kind, ready=False, stop_on_run=True
    )

    agent.learn(1, progress_factory=lambda *_args, **_kwargs: [])

    assert runtime.events[0].is_set()
    assert ("runtime.wait", ("job",), 300) in runtime.calls
    assert runtime.calls.count("logger.close") == 1
    assert runtime.calls.count("runtime.shutdown") == 1
    assert runtime.calls.index(("save_params", "/tmp/run")) < runtime.calls.index("logger.close")
    assert runtime.calls.index("logger.close") < runtime.calls.index("runtime.shutdown")


@pytest.mark.parametrize("module_name,class_name,kind", FAMILIES)
def test_distributed_learn_cleans_up_after_training_exception(
    monkeypatch, module_name, class_name, kind
):
    agent, runtime = _make_agent(
        monkeypatch, module_name, class_name, kind, ready=True, stop_on_run=False
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("train failed")

    agent.train_step = fail

    with pytest.raises(RuntimeError, match="train failed"):
        agent.learn(1, progress_factory=lambda *_args, **_kwargs: [1])

    assert runtime.events[0].is_set()
    assert ("runtime.wait", ("job",), 300) in runtime.calls
    assert runtime.calls.count("logger.close") == 1
    assert runtime.calls.count("runtime.shutdown") == 1


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
    monkeypatch.setattr(distributed_runtime.mp, "get_context", lambda: Context())

    with pytest.raises(RuntimeError, match="manager failed"):
        distributed_runtime.RayDistributedRuntime(num_cpus=3, num_gpus=0)

    assert calls == [
        (
            "ray.init",
            {"num_cpus": 3, "num_gpus": 0, "ignore_reinit_error": True},
        ),
        ("ray.shutdown",),
    ]
