from pprint import pprint

import jax


def print_param(name, params):
    if name:
        print(name)
    pprint(jax.tree_util.tree_map(lambda x: x.shape, params), sort_dicts=False)
