from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple


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
