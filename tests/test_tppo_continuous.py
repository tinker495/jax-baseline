"""Continuous TPPO distribution-shape and KL regressions."""

from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.TPPO.impala_tppo import IMPALA_TPPO
from jax_baselines.TPPO.tppo import TPPO


def _continuous_agent(cls, family):
    agent = cls.__new__(cls)
    agent.action_type = "continuous"
    agent.gamma = 0.99
    agent.lamda = 0.95
    agent.gae_normalize = False
    agent.gae_normalize_scope = "batch"
    agent.preproc = lambda params, key, obses: obses[0]
    agent.critic = lambda params, key, feature: jnp.zeros((*feature.shape[:-1], 1))
    agent.actor = lambda params, key, feature: (
        jnp.zeros((*feature.shape[:-1], 2)),
        jnp.zeros((1, 2)),
    )
    agent.get_logprob = MethodType(family.get_logprob_continuous, agent)
    return agent


def _rollout(worker_count=2, timesteps=3):
    obses = [[np.zeros((timesteps, 3), dtype=np.float32)] for _ in range(worker_count)]
    actions = [np.zeros((timesteps, 2), dtype=np.float32) for _ in range(worker_count)]
    scalars = [np.zeros((timesteps, 1), dtype=np.float32) for _ in range(worker_count)]
    return obses, actions, scalars


def test_local_tppo_preprocess_preserves_and_flattens_gaussian_tuple():
    agent = _continuous_agent(TPPO, Actor_Critic_Policy_Gradient_Family)
    obses, actions, scalars = _rollout()

    out = TPPO._preprocess(
        agent,
        None,
        None,
        obses,
        actions,
        scalars,
        obses,
        scalars,
        scalars,
    )

    old_prob = out[4]
    assert isinstance(old_prob, tuple)
    assert old_prob[0].shape == (6, 2)
    assert old_prob[1].shape == (6, 2)


def test_impala_tppo_preprocess_preserves_and_flattens_gaussian_tuple():
    agent = _continuous_agent(IMPALA_TPPO, IMPALA_Family)
    agent.mu_ratio = 0.0
    agent._compute_vtrace = lambda pi, mu, reward, term, trunc, value, next_value: (
        jnp.zeros_like(pi),
        jnp.ones_like(pi),
        jnp.zeros_like(pi),
    )
    obses, actions, scalars = _rollout()

    out = IMPALA_TPPO.preprocess(
        agent,
        None,
        None,
        obses,
        actions,
        scalars,
        scalars,
        obses,
        scalars,
        scalars,
    )

    old_prob = out[3]
    assert isinstance(old_prob, tuple)
    assert old_prob[0].shape == (6, 2)
    assert old_prob[1].shape == (6, 2)


def _assert_continuous_loss_is_finite(agent, loss):
    agent.val_coef = 0.2
    agent.ent_coef = 0.01
    agent.use_entropy_adv_shaping = False
    agent.kl_coef = 5.0
    agent.kl_range = 0.05
    batch = 2
    old_prob = (jnp.zeros((batch, 2)), jnp.zeros((batch, 2)))
    args = [
        None,
        [jnp.zeros((batch, 3))],
        jnp.zeros((batch, 2)),
        jnp.zeros((batch, 1)),
    ]
    if loss is TPPO._loss_continuous:
        args.append(jnp.zeros((batch, 1)))
    args.extend(
        [
            old_prob,
            jnp.zeros((batch, 1)),
            jnp.ones((batch, 1)),
            None,
        ]
    )
    total, aux = loss(agent, *args)
    assert jnp.isfinite(total)
    assert jnp.all(jnp.isfinite(jnp.asarray(aux)))


def test_local_tppo_continuous_loss_broadcasts_log_std_for_kl():
    agent = _continuous_agent(TPPO, Actor_Critic_Policy_Gradient_Family)
    agent.value_clip = 0.3
    _assert_continuous_loss_is_finite(agent, TPPO._loss_continuous)


def test_impala_tppo_continuous_loss_converts_log_std_to_std_for_kl():
    agent = _continuous_agent(IMPALA_TPPO, IMPALA_Family)
    _assert_continuous_loss_is_finite(agent, IMPALA_TPPO._loss_continuous)


@pytest.mark.parametrize(
    ("cls", "family", "kind"),
    [
        (TPPO, Actor_Critic_Policy_Gradient_Family, "local"),
        (IMPALA_TPPO, IMPALA_Family, "impala"),
    ],
)
def test_continuous_tppo_full_train_step_handles_gaussian_minibatches(cls, family, kind):
    agent = _continuous_agent(cls, family)
    agent.params = {"bias": jnp.asarray(0.0, dtype=jnp.float32)}
    agent.preproc = lambda params, key, obses: obses[0]
    agent.critic = lambda params, key, feature: (
        jnp.zeros((*feature.shape[:-1], 1)) + params["bias"]
    )
    agent.actor = lambda params, key, feature: (
        jnp.zeros((*feature.shape[:-1], 2)) + params["bias"],
        jnp.zeros((1, 2)),
    )
    agent.optimizer = optax.sgd(1e-3)
    agent.opt_state = agent.optimizer.init(agent.params)
    agent.val_coef = 0.2
    agent.ent_coef = 0.01
    agent.use_entropy_adv_shaping = False
    agent.kl_coef = 5.0
    agent.kl_range = 0.05
    agent.minibatch_size = 2
    agent.epoch_num = 1
    agent._loss = MethodType(cls._loss_continuous, agent)
    obses, actions, scalars = _rollout(timesteps=2)

    if kind == "local":
        agent.gae_normalize = False
        agent.gae_normalize_scope = "batch"
        agent.value_clip = 0.3
        result = TPPO._train_step(
            agent,
            agent.params,
            agent.opt_state,
            jax.random.PRNGKey(0),
            obses,
            actions,
            scalars,
            obses,
            scalars,
            scalars,
        )
    else:
        agent.mu_ratio = 0.0
        agent.rho_max = 1.0
        agent.cut_max = 1.0
        result = IMPALA_TPPO._train_step(
            agent,
            agent.params,
            agent.opt_state,
            jax.random.PRNGKey(0),
            obses,
            actions,
            scalars,
            scalars,
            obses,
            scalars,
            scalars,
        )

    assert jnp.isfinite(result[0]["bias"])
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in result[2:])
