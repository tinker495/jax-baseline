import pytest

from model_builder.flax.apply import get_apply_fn_flax_module


@pytest.mark.parametrize(
    "key,method", [(None, None), ("key", None), (None, "method"), ("key", "method")]
)
def test_flax_apply_forwards_only_requested_options(key, method):
    class Module:
        def apply(self, *args, **kwargs):
            self.call = args, kwargs
            return "result"

    module = Module()
    apply_fn = get_apply_fn_flax_module(module, method=method, mutable=["state"])

    assert apply_fn("params", key, "input") == "result"
    assert module.call == (
        ("params", "input"),
        {
            "mutable": ["state"],
            **({"rngs": {"params": key}} if key is not None else {}),
            **({"method": method} if method is not None else {}),
        },
    )
