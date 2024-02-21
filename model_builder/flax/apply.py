import flax.linen as nn


def get_apply_fn_flax_module(module: nn.Module, method: nn.Module.__call__ = None):
    if method is None:

        def apply_fn(params, key, *x):
            if key is None:
                return module.apply(params, *x)
            else:
                return module.apply(params, *x, rngs={"params": key})

    else:

        def apply_fn(params, key, *x):
            if key is None:
                return module.apply(params, *x, method=method)
            else:
                return module.apply(params, *x, rngs={"params": key}, method=method)

    return apply_fn
