"""Coverage for the hparam-provider protocol in jax_baselines.core.hparams.

``get_hyper_params`` reflects scalar config off the agent, then discovers held
providers exposing a callable ``hparams() -> dict`` and merges their dicts. This
is what logs ``self.ckpt.hparams()`` without the gatherer naming "ckpt", and
what lets future deep handles log for free.
"""

from jax_baselines.core.hparams import get_hyper_params


class _Provider:
    def hparams(self):
        return {"baseline_mode": "median", "baseline_q": 0.2}


class _NotAProvider:
    """A held object that does not implement the provider protocol."""

    value = 7


class _Agent:
    def __init__(self):
        self.learning_rate = 5e-5  # plain scalar -> reflected
        self.optimizer = "adamw"  # plain scalar -> reflected
        self.ckpt = _Provider()  # provider -> merged
        self.other = _NotAProvider()  # no hparams() -> ignored
        self._private = "hidden"  # underscore -> skipped


def test_scalar_attrs_are_reflected():
    params = get_hyper_params(_Agent())
    assert params["learning_rate"] == 5e-5
    assert params["optimizer"] == "adamw"


def test_provider_hparams_are_merged():
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


def test_provider_merge_is_generic_not_named_ckpt():
    """A provider held under any attribute name is merged."""

    class _AltAgent:
        def __init__(self):
            self.optimizer_handle = _Provider()  # not named "ckpt"

    params = get_hyper_params(_AltAgent())
    assert params["baseline_mode"] == "median"
    assert params["baseline_q"] == 0.2
