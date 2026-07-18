import numpy as np


def dummy_observation(space):
    return {key: np.zeros((1, *shape), dtype=np.float32) for key, shape in space.items()}


def print_flax_model_summary(enabled, key, *models):
    if not enabled:
        return

    for model, *inputs in models:
        print(model.tabulate(key, *inputs))


def print_haiku_model_summary(enabled, *models):
    if not enabled:
        return

    import haiku as hk

    for model, *inputs in models:
        print(hk.experimental.tabulate(model)(*inputs))
