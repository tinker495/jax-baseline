"""Coverage for scalar reflection in jax_baselines.core.hparams.

``get_hyper_params`` reflects public scalar config directly off the agent and
ignores nested runtime objects.
"""

from jax_baselines.core.hparams import get_hyper_params


class _NotAProvider:
    """A held object that does not implement the provider protocol."""

    value = 7


class _Agent:
    def __init__(self):
        self.learning_rate = 5e-5  # plain scalar -> reflected
        self.optimizer = "adamw"  # plain scalar -> reflected
        self.baseline_mode = "median"
        self.baseline_q = 0.2
        self.other = _NotAProvider()  # no hparams() -> ignored
        self._private = "hidden"  # underscore -> skipped


def test_scalar_attrs_are_reflected():
    params = get_hyper_params(_Agent())
    assert params["learning_rate"] == 5e-5
    assert params["optimizer"] == "adamw"


def test_checkpoint_hparams_are_reflected_from_agent():
    params = get_hyper_params(_Agent())
    assert params["baseline_mode"] == "median"
    assert params["baseline_q"] == 0.2


def test_non_provider_attr_is_ignored():
    params = get_hyper_params(_Agent())
    # _NotAProvider has no hparams(); its presence contributes nothing extra.
    assert "value" not in params


def test_underscore_attrs_are_skipped():
    params = get_hyper_params(_Agent())
    assert "_private" not in params


def test_nested_objects_are_not_inspected():
    class _Provider:
        def hparams(self):
            return {"hidden_nested_value": 1}

    class _AltAgent:
        handle = _Provider()

    assert "hidden_nested_value" not in get_hyper_params(_AltAgent())
