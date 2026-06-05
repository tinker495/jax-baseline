import jax
import jax.numpy as jnp


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
        f,
        jnp.zeros((1,), dtype=jnp.float32),
        (deltas, terminateds, truncateds),
        reverse=True,
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
    return A + values
