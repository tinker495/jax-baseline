import flax.linen as nn


def get_apply_fn_flax_module(
    module: nn.Module, method: nn.Module.__call__ = None, mutable: list[str] = False
):
    if method is None:

        def apply_fn(params, key, *x):
            if key is None:
                return module.apply(params, *x, mutable=mutable)
            else:
                return module.apply(params, *x, rngs={"params": key}, mutable=mutable)

    else:

        def apply_fn(params, key, *x):
            if key is None:
                return module.apply(params, *x, method=method, mutable=mutable)
            else:
                return module.apply(
                    params, *x, rngs={"params": key}, method=method, mutable=mutable
                )

    return apply_fn
