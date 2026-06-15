import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.optimizers import make_optimizer_factory, select_optimizer


def _params():
    return {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}


def _grads():
    return {"w": jnp.array([0.25, -0.5], dtype=jnp.float32)}


@pytest.mark.parametrize(
    "name",
    [
        "adam",
        "nadam",
        "schedule_free_adam",
        "adopt",
        "nadopt",
        "adamw",
        "rmsprop",
        "sgd",
        "adabelief",
        "lion",
        "prodigy",
        "muon",
    ],
)
def test_select_optimizer_supported_registry_entries_update_tiny_tree(name):
    optimizer = select_optimizer(name, 0.1)
    state = optimizer.init(_params())
    updates, state = optimizer.update(_grads(), state, _params())

    leaves = jax.tree.leaves(updates)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


@pytest.mark.parametrize("name", ["ano", "normuon"])
def test_select_optimizer_contrib_registry_entries_work_when_available(monkeypatch, name):
    def fake_contrib_optimizer(lr, weight_decay=0.0):
        return optax.sgd(lr)

    monkeypatch.setattr(optax.contrib, name, fake_contrib_optimizer, raising=False)

    optimizer = select_optimizer(name, 0.1)
    state = optimizer.init(_params())
    updates, _ = optimizer.update(_grads(), state, _params())

    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(updates))


@pytest.mark.parametrize("name", ["ano", "normuon"])
def test_select_optimizer_contrib_registry_entries_raise_clearly_when_unavailable(
    monkeypatch, name
):
    monkeypatch.delattr(optax.contrib, name, raising=False)

    with pytest.raises(ValueError, match=f"requires optax.contrib.{name}"):
        select_optimizer(name, 0.1)


def test_select_optimizer_adam_path_matches_optax_reference_with_custom_eps():
    optimizer = select_optimizer("adam", 0.1, eps=1e-5)
    reference = optax.adam(0.1, b1=0.9, b2=0.999, eps=1e-5)

    updates, _ = optimizer.update(_grads(), optimizer.init(_params()), _params())
    expected, _ = reference.update(_grads(), reference.init(_params()), _params())

    np.testing.assert_allclose(updates["w"], expected["w"], rtol=1e-6)


def test_select_optimizer_grad_max_wraps_with_global_norm_clip():
    params = {"w": jnp.array([0.0, 0.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([3.0, 4.0], dtype=jnp.float32)}
    optimizer = select_optimizer("sgd", 1.0, grad_max=1.0)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)

    np.testing.assert_allclose(updates["w"], np.array([-0.6, -0.8], dtype=np.float32), rtol=1e-6)


def test_select_optimizer_reset_suffix_resets_inner_state_on_period_boundary():
    params = _params()
    optimizer = select_optimizer("adam_reset_2", 0.1)
    state = optimizer.init(params)

    _, state = optimizer.update(_grads(), state, params)
    assert int(state[1]) == 1

    _, state = optimizer.update(_grads(), state, params)
    fresh_inner_state = select_optimizer("adam", 0.1).init(params)
    assert int(state[1]) == 2
    for observed, expected in zip(jax.tree.leaves(state[0]), jax.tree.leaves(fresh_inner_state)):
        np.testing.assert_allclose(observed, expected, rtol=1e-6)


def test_select_optimizer_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        select_optimizer("not-an-optimizer", 0.1)


def test_make_optimizer_factory_captures_selection_policy():
    factory = make_optimizer_factory("adam", eps=1e-5, grad_max=0.5)
    optimizer = factory(0.1)
    reference = select_optimizer("adam", 0.1, eps=1e-5, grad_max=0.5)

    updates, _ = optimizer.update(_grads(), optimizer.init(_params()), _params())
    expected, _ = reference.update(_grads(), reference.init(_params()), _params())

    np.testing.assert_allclose(updates["w"], expected["w"], rtol=1e-6)
