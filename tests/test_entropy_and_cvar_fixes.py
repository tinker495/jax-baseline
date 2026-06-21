"""Regression locks for two owner-authorized 16th-pass fixes.

W1 — entropy_loss sign convention across the policy-gradient family.
    A2C/impala.py previously used `entropy_loss = jnp.mean(entropy_h)` (positive),
    the lone outlier among 16 PG loss bodies. On the non-default
    `use_entropy_adv_shaping=False` path `total_loss += ent_coef * entropy_loss`,
    so a positive sign turns the entropy BONUS into a PENALTY. Every site must use
    the negated mean so minimizing the loss maximizes entropy.

W3 — APE-X IQN risk-averse (CVaR) acting.
    The distributed worker `actor` must distort the acting quantiles by CVaR
    (mirroring local IQN._get_actions); the priority path get_abs_td_error must NOT
    (mirroring local iqn.py _loss/_target, which use plain U[0,1]).

These paths live inside loss closures / Ray-gated workers that cannot be built
without full models or a Ray runtime, so they are locked at the source level —
the same convention this suite uses for other distributed paths.
"""

import inspect
import re

from jax_baselines.A2C import a2c, impala
from jax_baselines.IQN import apex_iqn
from jax_baselines.PPO import impala_ppo, ppo
from jax_baselines.SPO import impala_spo, spo
from jax_baselines.TPPO import impala_tppo, tppo

# Every module that assigns `entropy_loss` from `entropy_h`.
PG_MODULES = [a2c, impala, ppo, impala_ppo, spo, impala_spo, tppo, impala_tppo]

_ENTROPY_H_ASSIGN = re.compile(r"entropy_loss\s*=\s*(-?)jnp\.mean\(entropy_h\)")


def test_w1_entropy_loss_sign_is_negated_everywhere():
    offenders = []
    for module in PG_MODULES:
        for lineno, line in enumerate(inspect.getsource(module).splitlines(), start=1):
            m = _ENTROPY_H_ASSIGN.search(line)
            if m and m.group(1) != "-":
                offenders.append(f"{module.__name__}:{lineno}: {line.strip()}")
    assert not offenders, (
        "entropy_loss must be `-jnp.mean(entropy_h)` (bonus, not penalty); "
        f"positive-sign offenders found: {offenders}"
    )


def test_w1_impala_ac_has_the_negated_form():
    # Guard against the specific regression: A2C/impala.py was the outlier.
    src = inspect.getsource(impala)
    assert "entropy_loss = -jnp.mean(entropy_h)" in src
    assert "entropy_loss = jnp.mean(entropy_h)" not in src


def test_w3_apex_iqn_actor_applies_cvar_priority_path_does_not():
    src = inspect.getsource(apex_iqn.APE_X_IQN.get_actor_builder)
    # CVaR captured into the builder closure.
    assert re.search(r"CVaR\s*=\s*self\.CVaR", src), "get_actor_builder must capture self.CVaR"

    # The acting `actor` distorts tau by CVaR.
    actor_src = src[
        src.index("def actor(") : src.index("def get_action")
        if "def get_action" in src
        else len(src)
    ]
    assert (
        "jax.random.uniform(key, (1, n_support)) * CVaR" in actor_src
    ), "apex IQN actor must scale acting tau by CVaR (mirror local IQN._get_actions)"

    # The priority path keeps plain U[0,1] (no CVaR), mirroring local iqn.py _loss/_target.
    abs_err_src = src[src.index("def get_abs_td_error(") : src.index("def actor(")]
    assert (
        "* CVaR" not in abs_err_src and "*CVaR" not in abs_err_src
    ), "get_abs_td_error must NOT apply CVaR (matches local IQN priority/TD path)"
