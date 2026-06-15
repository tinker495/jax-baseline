import jax
import jax.numpy as jnp


def discount_with_terminated(rewards, terminateds, truncateds, next_values, gamma):
    def f(ret, info):
        reward, term, trunc, nextval = info
        # done marks the episode boundary (terminated OR truncated); the return
        # accumulation resets there. At a boundary the value is bootstrapped from
        # nextval only on truncation (term == 0), never on a true terminal.
        done = term + trunc - term * trunc
        ret = reward + gamma * (ret * (1.0 - done) + nextval * (1.0 - term) * done)
        return ret, ret

    truncateds = truncateds.at[-1].set(jnp.ones((1,), dtype=jnp.float32))
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
        f,
        jnp.zeros((1,), dtype=jnp.float32),
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
    return A + values
