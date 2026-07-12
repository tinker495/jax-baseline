from typing import get_type_hints

import pytest

from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.distributed_runtime import DistributedRuntime
from jax_baselines.IMPALA.base_class import IMPALA_Family


class _Runtime:
    replay_manager = (
        create_event
    ) = create_param_server = create_impala_buffer = lambda self, *a: None
    create_worker = create_logger_server = wait = lambda self, *a, **kw: None

    def worker_info(self, worker):
        return worker

    def shutdown(self):
        return None


def test_distributed_runtime_is_an_explicit_structural_protocol():
    assert isinstance(_Runtime(), DistributedRuntime)


@pytest.mark.parametrize(
    "family", [Ape_X_Family, Ape_X_Deteministic_Policy_Gradient_Family, IMPALA_Family]
)
def test_distributed_families_depend_on_the_runtime_protocol(family):
    assert get_type_hints(family.__init__)["runtime"] is DistributedRuntime
