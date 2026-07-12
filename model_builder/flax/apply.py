from typing import Callable, Optional, Union

import flax.linen as nn


def get_apply_fn_flax_module(
    module: nn.Module,
    method: Optional[Callable] = None,
    mutable: Union[bool, list[str]] = False,
):
    def apply_fn(params, key, *x):
        kwargs = {"mutable": mutable}
        if key is not None:
            kwargs["rngs"] = {"params": key}
        if method is not None:
            kwargs["method"] = method
        return module.apply(params, *x, **kwargs)

    return apply_fn
