import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.optim import adopt, optimizer_reset_by_period, scale_by_adopt


def _params():
    return {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}


def _grads():
    return {"w": jnp.array([0.25, -0.5], dtype=jnp.float32)}


def test_scale_by_adopt_matches_locked_three_step_updates():
    optimizer = scale_by_adopt()
    state = optimizer.init(_params())

    observed = []
    for _ in range(3):
        updates, state = optimizer.update(_grads(), state, _params())
        observed.append(np.asarray(updates["w"]))

    np.testing.assert_allclose(observed[0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        observed[1], np.array([0.10000002, -0.10000002], dtype=np.float32), rtol=1e-6
    )
    np.testing.assert_allclose(
        observed[2], np.array([0.19000004, -0.19000004], dtype=np.float32), rtol=1e-6
    )
    assert int(state.count) == 3
    np.testing.assert_allclose(state.nu["w"], np.array([0.0625, 0.25], dtype=np.float32))
    np.testing.assert_allclose(
        state.mu["w"], np.array([0.19000004, -0.19000004], dtype=np.float32), rtol=1e-6
    )


def test_adopt_applies_learning_rate_with_negative_update_direction():
    optimizer = adopt(0.1)
    state = optimizer.init(_params())

    observed = []
    for _ in range(3):
        updates, state = optimizer.update(_grads(), state, _params())
        observed.append(np.asarray(updates["w"]))

    np.testing.assert_allclose(observed[0], np.array([-0.0, -0.0], dtype=np.float32))
    np.testing.assert_allclose(observed[1], np.array([-0.01, 0.01], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(observed[2], np.array([-0.019, 0.019], dtype=np.float32), rtol=1e-6)


def test_optimizer_reset_by_period_resets_state_after_boundary():
    optimizer = optimizer_reset_by_period(optax.trace(decay=0.9), reset_steps=2)
    state = optimizer.init(_params())

    updates, state = optimizer.update(_grads(), state, _params())
    np.testing.assert_allclose(updates["w"], np.array([0.25, -0.5], dtype=np.float32))
    np.testing.assert_allclose(state[0].trace["w"], np.array([0.25, -0.5], dtype=np.float32))
    assert int(state[1]) == 1

    updates, state = optimizer.update(_grads(), state, _params())
    np.testing.assert_allclose(updates["w"], np.array([0.475, -0.95], dtype=np.float32))
    np.testing.assert_allclose(state[0].trace["w"], np.array([0.0, 0.0], dtype=np.float32))
    assert int(state[1]) == 2

    updates, state = optimizer.update(_grads(), state, _params())
    np.testing.assert_allclose(updates["w"], np.array([0.25, -0.5], dtype=np.float32))
    np.testing.assert_allclose(state[0].trace["w"], np.array([0.25, -0.5], dtype=np.float32))
    assert int(state[1]) == 3
