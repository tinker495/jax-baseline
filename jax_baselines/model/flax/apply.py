import flax.linen as nn


def get_apply_fn_flax_module(module: nn.Module):
    def apply_fn(params, key, *x):
        if key is None:
            return module.apply(params, *x)
        else:
            return module.apply(params, *x, rngs={"noisy": key})

    return apply_fn
