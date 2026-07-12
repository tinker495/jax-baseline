from typing import Any, Protocol


class CheckpointStore(Protocol):
    def save(self, path: str, state: Any) -> None:
        ...

    def restore(self, path: str) -> Any:
        ...


class NoOpCheckpointStore:
    """Default for direct core use when no artifact adapter is injected."""

    def save(self, path: str, state: Any) -> None:
        return None

    def restore(self, path: str) -> Any:
        raise FileNotFoundError("No checkpoint store was supplied")


def checkpoint_store_or_default(store: CheckpointStore | None) -> CheckpointStore:
    return store if store is not None else NoOpCheckpointStore()
