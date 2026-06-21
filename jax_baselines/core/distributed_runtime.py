from __future__ import annotations

from typing import Any, Protocol


class DistributedEvent(Protocol):
    def set(self) -> None:
        ...

    def clear(self) -> None:
        ...

    def is_set(self) -> bool:
        ...


class DistributedWorkerHandle(Protocol):
    def get_info(self) -> dict:
        ...

    def run(self, *args, **kwargs) -> Any:
        ...


class ParamServerHandle(Protocol):
    def get_params(self) -> Any:
        ...

    def update_params(self, params) -> Any:
        ...


class LoggerServerHandle(Protocol):
    def get_log_dir(self) -> str:
        ...

    def add_multiline(self, eps) -> Any:
        ...

    def log_trainer(self, step, log_dict) -> Any:
        ...

    def log_worker(self, log_dict, episode) -> Any:
        ...

    def last_update(self) -> Any:
        ...

    def close(self) -> Any:
        ...

    def register_hparams(self, hparams: dict) -> Any:
        ...


class ImpalaRolloutBuffer(Protocol):
    def queue_info(self) -> tuple:
        ...

    def queue_is_empty(self) -> bool:
        ...

    def sample(self) -> Any:
        ...

    def clear(self) -> None:
        ...


class DistributedRuntime(Protocol):
    def replay_manager(self) -> Any:
        ...

    def create_event(self) -> DistributedEvent:
        ...

    def worker_info(self, worker: DistributedWorkerHandle) -> dict:
        ...

    def create_param_server(self, params) -> ParamServerHandle:
        ...

    def create_logger_server(
        self, log_dir, run_name, experiment_name="experiment", logger_factory=None
    ) -> LoggerServerHandle:
        ...

    def create_impala_buffer(
        self,
        replay_size: int,
        actor_num: int,
        observation_space: list,
        discrete=True,
        action_space=1,
        sample_size=32,
        seed=None,
    ) -> ImpalaRolloutBuffer:
        ...

    def wait(self, jobs, timeout=None) -> Any:
        ...

    def shutdown(self) -> None:
        ...
