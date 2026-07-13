import jax.numpy as jnp
import pytest

from jax_baselines.C51.c51 import C51
from jax_baselines.C51.hl_gauss_c51 import HL_GAUSS_C51
from jax_baselines.SPR.spr import SPR

PREDICTIONS = jnp.asarray([[[0.9, 0.1]], [[0.8, 0.2]]])
TARGETS = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
ACTIONS = jnp.zeros((2, 1, 1), dtype=jnp.int32)


@pytest.mark.parametrize("family", (C51, HL_GAUSS_C51))
def test_c51_loss_applies_per_importance_weights(family):
    agent = family.__new__(family)
    agent.get_q = lambda *_args: PREDICTIONS

    full, _ = family._loss(agent, None, None, ACTIONS, TARGETS, jnp.ones(2), None)
    first_only, per_sample = family._loss(
        agent, None, None, ACTIONS, TARGETS, jnp.asarray([1.0, 0.0]), None
    )

    assert first_only == pytest.approx(float(per_sample[0] / 2))
    assert first_only != pytest.approx(float(full))


def test_spr_loss_applies_per_importance_weights():
    agent = SPR.__new__(SPR)
    agent.get_q = lambda *_args: PREDICTIONS
    agent._represetation_loss = lambda *_args: jnp.asarray(0.0)
    agent.spr_weight = 5.0

    def loss(weights):
        return SPR._loss(
            agent,
            None,
            None,
            None,
            None,
            None,
            None,
            ACTIONS,
            TARGETS,
            weights,
            None,
        )[0]

    assert loss(jnp.asarray([1.0, 0.0])) != pytest.approx(float(loss(jnp.ones(2))))
