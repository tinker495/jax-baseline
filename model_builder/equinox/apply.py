import equinox as eqx

from model_builder.equinox.Module import apply_batch_stats, extract_batch_stats, has_batch_stats


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


def _call_with_optional_key(fn, key, *args, **kwargs):
    if key is None:
        return fn(*args, **kwargs)
    try:
        return fn(*args, key=key, **kwargs)
    except TypeError:
        return fn(*args, **kwargs)


def get_apply_fn_equinox_module(static_module, method=_SENTINEL):
    method_name = None if method is _SENTINEL else method.__name__

    def apply_fn(params, key, *args, **kwargs):
        module_params = params
        batch_stats = None
        if isinstance(params, dict) and "params" in params:
            module_params = params["params"]
            batch_stats = params.get("batch_stats")
        module = eqx.combine(module_params, static_module)
        module = apply_batch_stats(module, batch_stats)
        if method_name is None:
            fn = module
        else:
            fn = getattr(module, method_name)
        result = _call_with_optional_key(fn, key, *args, **kwargs)
        if isinstance(result, tuple):
            maybe_module = result[-1]
            if isinstance(maybe_module, eqx.Module):
                output = result[:-1]
                if len(output) == 1:
                    output = output[0]
                updated_params, _ = eqx.partition(maybe_module, eqx.is_array)
                updates = {}
                batch_stats_update = extract_batch_stats(updated_params)
                if has_batch_stats(batch_stats_update):
                    updates["batch_stats"] = batch_stats_update
                return output, updates
        return result

    return apply_fn
