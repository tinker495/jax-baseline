import jax
import jax.numpy as jnp


def discount_with_terminated(rewards, terminateds, truncateds, next_values, gamma):
    # rewards/terminateds/truncateds arrive flat ([T]) from the scalar-per-step
    # buffer, while the critic's next_values keep a trailing unit dim ([T, 1]).
    # Reshape so the scan stays [T, 1] (otherwise truncateds[-1] is a scalar and
    # the boundary-set below fails to broadcast). Mirrors get_gaes' flat->[T,1]
    # alignment.
    rewards = rewards.reshape(next_values.shape)
    terminateds = terminateds.reshape(next_values.shape).astype(bool)
    truncateds = truncateds.reshape(next_values.shape).astype(bool)

    def f(ret, info):
        reward, term, trunc, nextval = info
        # done marks the episode boundary (terminated OR truncated); the return
        # accumulation resets there. At a boundary the value is bootstrapped from
        # nextval only on truncation (term == 0), never on a true terminal.
        done = jnp.logical_or(term, trunc)
        bootstrap = jnp.where(term, jnp.zeros_like(nextval), nextval)
        ret = reward + gamma * jnp.where(done, bootstrap, ret)
        return ret, ret

    truncateds = truncateds.at[-1].set(jnp.ones((1,), dtype=bool))
    _, discounted = jax.lax.scan(
        f,
        jnp.zeros((1,), dtype=jnp.float32),
        (rewards, terminateds, truncateds, next_values),
        reverse=True,
    )
    return discounted


def get_gaes(rewards, terminateds, truncateds, values, next_values, gamma, lamda):
    # rewards/terminateds/truncateds arrive flat ([T]) from the scalar-per-step
    # buffer, while the critic's values/next_values keep a trailing unit dim
    # ([T, 1]). Align them so deltas stays [T, 1]; otherwise [T] broadcasts
    # against [T, 1] into a [T, T] matrix and the scan carry shape blows up.
    rewards = rewards.reshape(values.shape)
    terminateds = terminateds.reshape(values.shape).astype(bool)
    truncateds = truncateds.reshape(values.shape).astype(bool)
    bootstrap = jnp.where(terminateds, jnp.zeros_like(next_values), next_values)
    deltas = rewards + gamma * bootstrap - values

    def f(last_gae_lam, info):
        delta, term, trunc = info
        boundary = jnp.logical_or(term, trunc)
        continuation = jnp.where(boundary, jnp.zeros_like(last_gae_lam), last_gae_lam)
        last_gae_lam = delta + gamma * lamda * continuation
        return last_gae_lam, last_gae_lam

    _, advs = jax.lax.scan(
        f,
        jnp.zeros_like(deltas[0]),
        (deltas, terminateds, truncateds),
        reverse=True,
    )
    return advs


ADVANTAGE_NORMALIZE_SCOPES = ("batch", "minibatch")


def normalize_advantage(adv):
    """Standardize advantages to zero mean / unit std (eps-guarded).

    Reduces over every element of ``adv``, so the scope is set by what is passed
    in: the whole flattened rollout gives batch-scope normalization (once per
    update), a single minibatch gives minibatch-scope normalization (PPO2-style,
    recomputed per minibatch per epoch).
    """
    return (adv - jnp.mean(adv, keepdims=True)) / (jnp.std(adv, keepdims=True) + 1e-6)


def validate_advantage_normalize_scope(scope):
    if scope not in ADVANTAGE_NORMALIZE_SCOPES:
        raise ValueError(
            f"gae_normalize_scope must be one of {ADVANTAGE_NORMALIZE_SCOPES}, got {scope!r}"
        )
    return scope


def get_vtrace(rewards, rhos, c_ts, terminateds, truncateds, values, next_values, gamma):
    terminateds = terminateds.astype(bool)
    truncateds = truncateds.astype(bool)
    bootstrap = jnp.where(terminateds, jnp.zeros_like(next_values), next_values)
    deltas = rhos * (rewards + gamma * bootstrap - values)

    def f(last_v, info):
        delta, c_t, term, trunc = info
        boundary = jnp.logical_or(term, trunc)
        continuation = jnp.where(boundary, jnp.zeros_like(last_v), last_v)
        last_v = delta + gamma * c_t * continuation
        return last_v, last_v

    _, A = jax.lax.scan(
        f,
        jnp.zeros((1,), dtype=jnp.float32),
        (deltas, c_ts, terminateds, truncateds),
        reverse=True,
    )
    return A + values
