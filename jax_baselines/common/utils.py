import os
import pickle
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

cpu_jit = partial(jax.jit, backend="cpu")
gpu_jit = partial(jax.jit, backend="gpu")

PyTree = Any


def save(ckpt_dir: str, obs) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(obs):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_util.tree_map(lambda t: 0, obs)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree.flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree.unflatten(treedef, flat_state)


def key_gen(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree = None, treedef=None):
    if treedef is None:
        treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree, mul=1.2):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t) * mul,
        target,
        keys_tree,
    )


def scaled_by_reset(
    tensors: PyTree,
    optimizer_state: optax.GradientTransformationExtraArgs,
    optimizer: optax.GradientTransformation,
    key: jax.random.PRNGKey,
    steps: int,
    update_period: int,
    tau: float,
):
    update = steps % update_period == 0

    def _soft_reset(old_tensors, old_optimizer_state, key):
        new_tensors = tree_random_normal_like(key, tensors)
        soft_reseted = jax.tree_util.tree_map(
            lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
        )
        # name dense is hardreset
        return soft_reseted, optimizer.init(soft_reseted)

    tensors, optimizer_state = jax.lax.cond(
        update, _soft_reset, lambda p, os, k: (p, os), tensors, optimizer_state, key
    )
    return tensors, optimizer_state


def scaled_by_reset_with_filter(
    tensors: PyTree,
    optimizer_state: optax.GradientTransformationExtraArgs,
    optimizer: optax.GradientTransformation,
    key: jax.random.PRNGKey,
    steps: int,
    update_period: int,
    taus: PyTree,
):
    update = steps % update_period == 0

    def _soft_reset(old_tensors, old_optimizer_state, key):
        new_tensors = tree_random_normal_like(key, tensors)
        soft_reseted = jax.tree_util.tree_map(
            lambda new, old, tau: tau * new + (1.0 - tau) * old, new_tensors, old_tensors, taus
        )
        # name dense is hardreset
        return soft_reseted, optimizer.init(soft_reseted)

    tensors, optimizer_state = jax.lax.cond(
        update, _soft_reset, lambda p, os, k: (p, os), tensors, optimizer_state, key
    )
    return tensors, optimizer_state


def filter_like_tree(tensors: PyTree, name_filter: str, filter_fn: Callable):
    # name_filter: "qnet"
    # filter_fn: lambda x,filtered: jnp.ones_like(x) if filtered else jnp.ones_like(x) * 0.2
    # make a new tree with the same structure as the input tree, but with the values filtered by the filter_fn
    # for making tau = 1.0 for qnet and tau = 0.2 for the rest
    # this making hard_reset for qnet and scaled_by_reset for the rest
    def sigma_filter(x, sigma):
        return jnp.zeros_like(x) if sigma else x

    # noisynet's sigma is 0.0 for did not reset noisynet noise
    tensors = jax.tree_util.tree_map(lambda x: x, tensors)

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
    return jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(update, new, old), new_tensors, old_tensors
    )


def soft_update(new_tensors: PyTree, old_tensors: PyTree, tau: float):
    return jax.tree_util.tree_map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
    )


def truncated_mixture(quantiles, cut):
    """Concatenates and sorts quantile values, then truncates the highest values.

    Used in TQC and CrossQ_TQC algorithms to implement truncated quantile critics.

    Args:
        quantiles: List of quantile values from multiple critics to be mixed
        cut: Number of highest quantile values to remove

    Returns:
        Sorted and truncated quantile values with the highest 'cut' values removed
    """
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


def kl_divergence_discrete(p, q, eps: float = 1e-8):
    # Add epsilon to prevent log(0)
    p_safe = p + eps
    q_safe = q + eps
    # Compute log values
    log_p = jnp.log(p_safe)
    log_q = jnp.log(q_safe)
    # Compute KL divergence with masking
    return jnp.sum(p * (log_p - log_q))


def kl_divergence_continuous(p, q):
    p_mu, p_std = p
    q_mu, q_std = q
    term1 = jnp.log(q_std / p_std)
    term2 = (p_std**2 + (p_mu - q_mu) ** 2) / (2 * q_std**2)
    return jnp.sum(term1 + term2 - 0.5, axis=-1, keepdims=True)


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


def add_hparams(agent, tensorboardrun):
    hparam_dict = get_hyper_params(agent)
    tensorboardrun.log_param(hparam_dict)


def compute_ckpt_window_stat(
    returns_window: list, q: float, use_standardization: bool, mode: str = "quantile"
):
    """Compute robust window statistic for checkpoint decisions.

    Args:
        returns_window: List of episode returns in current window
        q: Quantile parameter (used when mode="quantile")
        use_standardization: Whether to standardize using median and MAD
        mode: Baseline computation mode - "min", "median", "quantile", or "mean"

    Returns:
        Window statistic value or None if window is empty
    """
    if returns_window is None or len(returns_window) == 0:
        return None
    arr = np.asarray(returns_window, dtype=np.float64)
    if use_standardization and arr.size >= 3:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        scale = mad if mad > 1e-8 else 1.0
        arr = (arr - med) / scale

    # Dispatch by mode
    if mode == "min":
        return float(np.min(arr))
    elif mode == "median":
        return float(np.median(arr))
    elif mode == "mean":
        return float(np.mean(arr))
    elif mode == "quantile" or mode == "lower_percent":
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(arr, q))
    else:
        # Default to quantile for unknown modes
        q = float(q)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(arr, q))


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shapes: list = [()], dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.means = [np.zeros(shape, dtype=dtype) for shape in shapes]
        self.vars = [np.ones(shape, dtype=dtype) for shape in shapes]
        self.count = epsilon

    def normalize(self, xs):
        """Normalizes the input using the running mean and variance."""
        return [(x - mean) / np.sqrt(var + 1e-8) for x, mean, var in zip(xs, self.means, self.vars)]

    def update(self, xs):
        """Updates the mean, var and count from a batch of samples."""
        means = []
        vars = []
        for x, mean, var in zip(xs, self.means, self.vars):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            mean, var = self.update_mean_var_count_from_moments(
                mean, var, batch_mean, batch_var, batch_count
            )
            means.append(mean)
            vars.append(var)
        self.means = means
        self.vars = vars
        self.count += batch_count

    def update_mean_var_count_from_moments(self, mean, var, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean

        tot_count = self.count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var
