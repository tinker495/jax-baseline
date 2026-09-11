import tempfile

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.checkpoint_store import FileCheckpointStore
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.APE_X.base_class import Ape_X_Family
from jax_baselines.APE_X.dpg_base_class import Ape_X_Deteministic_Policy_Gradient_Family
from jax_baselines.core.checkpoint_state import ACCheckpointState
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
    "family",
    [
        Actor_Critic_Policy_Gradient_Family,
        Ape_X_Family,
        Ape_X_Deteministic_Policy_Gradient_Family,
        IMPALA_Family,
    ],
)
def test_remaining_families_delegate_checkpoint_io(family):
    class MemoryStore:
        def __init__(self, restored):
            self.restored = restored

        def save(self, path, state):
            self.saved = (path, state)

        def restore(self, path):
            self.restored_path = path
            return self.restored

    agent = family.__new__(family)
    if family in (Actor_Critic_Policy_Gradient_Family, IMPALA_Family):
        agent.actor_params = {"weights": 1}
        agent.critic_params = {"weights": 2}
        if family is Actor_Critic_Policy_Gradient_Family:
            agent.obs_rms = None
            agent.memory_backend = "cpu"
            agent.memory_device = None
        saved = ACCheckpointState(
            actor_params=agent.actor_params, critic_params=agent.critic_params
        )
        restored = ACCheckpointState(actor_params={"weights": 3}, critic_params={"weights": 4})
    elif family is Ape_X_Deteministic_Policy_Gradient_Family:
        agent.policy_params = {"weights": 1}
        agent.critic_params = {"weights": 2}
        agent.target_policy_params = {"weights": 3}
        agent.target_critic_params = {"weights": 4}
        agent.optimizer = optax.sgd(1e-3)
        saved = {
            "policy": agent.policy_params,
            "critic": agent.critic_params,
            "target_policy": agent.target_policy_params,
            "target_critic": agent.target_critic_params,
        }
        restored = {
            "policy": {"weights": 5},
            "critic": {"weights": 6},
            "target_policy": {"weights": 7},
            "target_critic": {"weights": 8},
        }
    else:
        agent.params = {"weights": 1}
        saved = agent.params
        restored = {"weights": 2}
    agent.checkpoint_store = MemoryStore(restored)

    agent.save_params("checkpoint")
    agent.load_params("checkpoint")

    assert agent.checkpoint_store.saved == ("checkpoint", saved)
    assert agent.checkpoint_store.restored_path == "checkpoint"
    if family in (Actor_Critic_Policy_Gradient_Family, IMPALA_Family):
        assert agent.actor_params == {"weights": 3}
        assert agent.critic_params == {"weights": 4}
    elif family is Ape_X_Deteministic_Policy_Gradient_Family:
        assert agent.policy_params == {"weights": 5}
        assert agent.critic_params == {"weights": 6}
        assert agent.target_policy_params == {"weights": 7}
        assert agent.target_critic_params == {"weights": 8}
    else:
        assert agent.params == {"weights": 2}
        assert agent.target_params is agent.params
