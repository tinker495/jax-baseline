import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.checkpoint_store import FileCheckpointStore
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.checkpoint_store import NoOpCheckpointStore
from jax_baselines.IMPALA.base_class import IMPALA_Family


def test_file_checkpoint_store_round_trip():
    state = {"weights": jnp.asarray([1.0, 2.0]), "step": np.asarray(3)}
    with tempfile.TemporaryDirectory() as directory:
        store = FileCheckpointStore()
        store.save(directory, state)
        restored = store.restore(directory)

    np.testing.assert_array_equal(restored["weights"], state["weights"])
    np.testing.assert_array_equal(restored["step"], state["step"])


def test_noop_checkpoint_store_does_not_write_and_cannot_restore(tmp_path):
    store = NoOpCheckpointStore()
    store.save(str(tmp_path / "params"), {"weights": 1})
    assert not (tmp_path / "params").exists()
    with pytest.raises(FileNotFoundError, match="No checkpoint store"):
        store.restore(str(tmp_path / "params"))


@pytest.mark.parametrize(
    ("family", "sets_target"),
    [
        (Actor_Critic_Policy_Gradient_Family, False),
        (Ape_X_Family, True),
        (Ape_X_Deteministic_Policy_Gradient_Family, True),
        (IMPALA_Family, True),
    ],
)
def test_remaining_families_delegate_checkpoint_io(family, sets_target):
    class MemoryStore:
        restored = {"weights": 2}

        def save(self, path, state):
            self.saved = (path, state)

        def restore(self, path):
            self.restored_path = path
            return self.restored

    agent = family.__new__(family)
    agent.params = {"weights": 1}
    agent.checkpoint_store = MemoryStore()

    agent.save_params("checkpoint")
    agent.load_params("checkpoint")

    assert agent.checkpoint_store.saved == ("checkpoint", {"weights": 1})
    assert agent.checkpoint_store.restored_path == "checkpoint"
    assert agent.params == {"weights": 2}
    if sets_target:
        assert agent.target_params is agent.params
