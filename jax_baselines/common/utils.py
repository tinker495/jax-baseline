from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

cpu_jit = partial(jax.jit, backend="cpu")
gpu_jit = partial(jax.jit, backend="gpu")

PyTree = Any


def key_gen(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree = None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t),
        target,
        keys_tree,
    )


def scaled_by_reset(
    tensors: PyTree, key: jax.random.PRNGKey, steps: int, update_period: int, taus: PyTree
):
    update = steps % update_period == 0

    def _soft_reset(old_tensors, key):
        new_tensors = tree_random_normal_like(key, tensors)
        soft_reseted = jax.tree_map(
            lambda new, old, tau: tau * new + (1.0 - tau) * old, new_tensors, old_tensors, taus
        )
        # name dense is hardreset
        return soft_reseted

    tensors = jax.lax.cond(update, _soft_reset, lambda x, k: x, tensors, key)
    return tensors


def filter_like_tree(tensors: PyTree, name_filter: str, filter_fn: Callable):
    # name_filter: "qnet"
    # filter_fn: lambda x,filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2
    # make a new tree with the same structure as the input tree, but with the values filtered by the filter_fn
    # for making tau = 1.0 for qnet and tau = 0.2 for the rest
    # this making hard_reset for qnet and scaled_by_reset for the rest
    def sigma_filter(x, sigma):
        return jnp.zeros_like(x) if sigma else x

    # noisynet's sigma is 0.0 for did not reset noisynet noise
    tensors = jax.tree_map(lambda x: x, tensors)

    def _filter_like_tree(tensors, filtered: bool):
        for name, value in tensors.items():
            if isinstance(value, dict):
                tensors[name] = _filter_like_tree(value, filtered or name_filter in name)
            else:
                tensors[name] = sigma_filter(filter_fn(value, filtered), "sigma" in name)
        return tensors

    return _filter_like_tree(tensors, False)


def hard_update(new_tensors: PyTree, old_tensors: PyTree, steps: int, update_period: int):
    update = steps % update_period == 0
    return jax.tree_map(lambda new, old: jax.lax.select(update, new, old), new_tensors, old_tensors)


def soft_update(new_tensors: PyTree, old_tensors: PyTree, tau: float):
    return jax.tree_map(lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors)


def truncated_mixture(quantiles, cut):
    quantiles = jnp.concatenate(quantiles, axis=1)
    sorted = jnp.sort(quantiles, axis=1)
    return sorted[:, :-cut]


@cpu_jit
def convert_states(obs: list):
    return [(o * 255.0).astype(np.uint8) if len(o.shape) >= 4 else o for o in obs]


def convert_jax(obs: list):
    return [jax.device_get(o).astype(jnp.float32) for o in obs]


def q_log_pi(q, entropy_tau):
    q_submax = q - jnp.max(q, axis=1, keepdims=True)
    logsum = jax.nn.logsumexp(q_submax / entropy_tau, axis=1, keepdims=True)
    tau_log_pi = q_submax - entropy_tau * logsum
    return q_submax, tau_log_pi


def discounted(rewards, gamma=0.99):  # lfilter([1],[1,-gamma],x[::-1])[::-1]
    _gamma = 1
    out = 0
    for r in rewards:
        out += r * _gamma
        _gamma *= gamma
    return out


def discount_with_terminated(rewards, terminateds, truncateds, next_values, gamma):
    def f(ret, info):
        reward, term, trunc, nextval = info
        ret = reward + gamma * (ret * (1.0 - trunc) + nextval * (1.0 - term) * trunc)
        return ret, ret

    truncateds.at[-1].set(jnp.ones((1,), dtype=jnp.float32))
    _, discounted = jax.lax.scan(
        f,
        jnp.zeros((1,), dtype=jnp.float32),
        (rewards, terminateds, truncateds, next_values),
        reverse=True,
    )
    return discounted


def get_gaes(rewards, terminateds, truncateds, values, next_values, gamma, lamda):
    deltas = rewards + gamma * (1.0 - terminateds) * next_values - values

    def f(last_gae_lam, info):
        delta, term, trunc = info
        last_gae_lam = delta + gamma * lamda * (1.0 - term) * (1.0 - trunc) * last_gae_lam
        return last_gae_lam, last_gae_lam

    _, advs = jax.lax.scan(
        f, jnp.zeros((1,), dtype=jnp.float32), (deltas, terminateds, truncateds), reverse=True
    )
    return advs


def get_vtrace(rewards, rhos, c_ts, terminateds, truncateds, values, next_values, gamma):
    deltas = rhos * (rewards + gamma * (1.0 - terminateds) * next_values - values)

    def f(last_v, info):
        delta, c_t, term, trunc = info
        last_v = delta + gamma * c_t * (1.0 - term) * (1.0 - trunc) * last_v
        return last_v, last_v

    _, A = jax.lax.scan(
        f,
        jnp.zeros((1,), dtype=jnp.float32),
        (deltas, c_ts, terminateds, truncateds),
        reverse=True,
    )
    v = A + values
    return v


def kl_divergence_discrete(p, q, eps: float = 2**-17):
    return p.dot(jnp.log(p + eps) - jnp.log(q + eps))


def kl_divergence_continuous(p, q):
    p_mu, p_std = p
    q_mu, q_std = q
    return p_std - q_std + (q_std**2 + (q_mu - p_mu) ** 2) / (2.0 * p_std**2) - 0.5


def get_hyper_params(agent):
    return dict(
        [
            (attr, getattr(agent, attr))
            for attr in dir(agent)
            if not callable(getattr(agent, attr))
            and not attr.startswith("__")
            and not attr.startswith("_")
            and isinstance(getattr(agent, attr), (int, float, str, bool))
        ]
    )


def add_hparams(agent, writer, metric_dict=dict()):
    from tensorboardX.summary import hparams

    hparam_dict = get_hyper_params(agent)
    # metric_dict = dict([m,None] for m in metric)
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
