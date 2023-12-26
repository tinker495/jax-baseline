import haiku as hk


def get_apply_fn_haiku_module(module: hk.Module):
    def apply_fn(params, key, *x):
        return module.apply(params, key, *x)

    return apply_fn
