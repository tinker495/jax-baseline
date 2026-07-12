from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol


class DistributedRuntime(Protocol):
    def replay_manager(self) -> Any:
        ...

    def create_event(self) -> Any:
        ...

    def create_worker(self, worker_cls: type, *args: Any, **kwargs: Any) -> Any:
        ...

    def worker_info(self, worker: Any) -> Any:
        ...

    def create_param_server(self, params: Any) -> Any:
        ...

    def create_logger_server(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def create_impala_buffer(self, need: "ImpalaRolloutNeed") -> Any:
        ...

    def wait(self, jobs: Any, timeout: float | None = None) -> Any:
        ...

    def shutdown(self) -> None:
        ...


@dataclass(frozen=True, kw_only=True)
class ImpalaRolloutNeed:
    replay_size: int
    actor_num: int
    observation_space: Any
    discrete: bool = True
    action_space: Any = 1
    sample_size: int = 32
    seed: Any = None


class ImpalaBatch(NamedTuple):
    obses: Any
    actions: Any
    mu_log_prob: Any
    rewards: Any
    nxtobses: Any
    terminateds: Any
    truncateds: Any


batch = ImpalaBatch
