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
    agent.critic = lambda params, key, obses: jnp.zeros((*obses["unified_obs"].shape[:-1], 1))
    agent.actor = lambda params, key, obses: (
        jnp.zeros((*obses["unified_obs"].shape[:-1], 2)),
        jnp.zeros((1, 2)),
    )
    agent.get_logprob = MethodType(family.get_logprob_continuous, agent)
    return agent


def _rollout(worker_count=2, timesteps=3):
    obses = [
        {"unified_obs": np.zeros((timesteps, 3), dtype=np.float32)} for _ in range(worker_count)
    ]
    actions = [np.zeros((timesteps, 2), dtype=np.float32) for _ in range(worker_count)]
    scalars = [np.zeros((timesteps, 1), dtype=np.float32) for _ in range(worker_count)]
    return obses, actions, scalars


@pytest.mark.parametrize(
    ("cls", "family", "kind"),
    [
        (TPPO, Actor_Critic_Policy_Gradient_Family, "local"),
        (IMPALA_TPPO, IMPALA_Family, "impala"),
    ],
)
def test_continuous_tppo_full_train_step_handles_gaussian_minibatches(cls, family, kind):
    agent = _continuous_agent(cls, family)
    agent.actor_params = {"bias": jnp.asarray(0.0, dtype=jnp.float32)}
    agent.critic_params = {"bias": jnp.asarray(0.0, dtype=jnp.float32)}
    agent.critic = lambda params, key, obses: (
        jnp.zeros((*obses["unified_obs"].shape[:-1], 1)) + params["bias"]
    )
    agent.actor = lambda params, key, obses: (
        jnp.zeros((*obses["unified_obs"].shape[:-1], 2)) + params["bias"],
        jnp.zeros((1, 2)),
    )
    agent.optimizer = optax.sgd(1e-3)
    agent.actor_opt_state = agent.optimizer.init(agent.actor_params)
    agent.critic_opt_state = agent.optimizer.init(agent.critic_params)
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
        obses = {"unified_obs": jnp.asarray([observation["unified_obs"] for observation in obses])}
        actions = jnp.asarray(actions)
        scalars = jnp.asarray(scalars).squeeze(-1)
        agent.gae_normalize = False
        agent.gae_normalize_scope = "batch"
        agent.value_clip = 0.3
        result = TPPO._train_step(
            agent,
            agent.actor_params,
            agent.critic_params,
            agent.actor_opt_state,
            agent.critic_opt_state,
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
            agent.actor_params,
            agent.critic_params,
            agent.actor_opt_state,
            agent.critic_opt_state,
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
    assert jnp.isfinite(result[1]["bias"])
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in result[4:])
