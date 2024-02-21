import jax


def formatData(t, s):
    if not isinstance(t, dict) and not isinstance(t, list):
        print(": " + str(t), end="")
    else:
        for key in t:
            print("\n" + "\t" * s + str(key), end="")
            if not isinstance(t, list):
                formatData(t[key], s + 1)


def print_param(name, params):
    if name:
        print(name, end="")
    param_tree_map = jax.tree_map(lambda x: x.shape, params)
    formatData(param_tree_map, 1 if name else 0)
    print()
