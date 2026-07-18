import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.cli.dpg import DPG_RUNNER
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.SAC.sac import (
    SAC,
    entropy_target_from_sigma,
    mode_action,
    sample_action,
)
from jax_baselines.TQC.tqc import TQC


def test_action_sampling_and_mode_are_explicitly_separate():
    mu = jnp.array([[0.2, -0.4]])
    log_std = jnp.zeros_like(mu)

    np.testing.assert_allclose(mode_action(mu), jnp.tanh(mu))
    assert not np.allclose(sample_action(mu, log_std, jax.random.PRNGKey(0)), mode_action(mu))


def test_dpg_eval_path_uses_sac_mode_without_sampling():
    agent = object.__new__(SAC)
    agent.simba = False
    agent.learning_starts = 100
    agent.use_checkpointing = False
    agent.policy_params = None
    agent.preproc = lambda params, key, obs: obs["obs"]
    agent.actor = lambda params, key, feature: (feature, jnp.full_like(feature, 10.0))

    obs = {"obs": jnp.array([[0.25, -0.5]])}
    actions = agent.actions(obs, steps=0, eval=True)

    np.testing.assert_allclose(actions, jnp.tanh(obs["obs"]))


def test_sac_actor_loss_uses_minimum_expected_q():
    agent = object.__new__(SAC)
    agent.preproc = lambda params, key, obs: obs
    agent._get_pi_log_prob = lambda params, feature, key: (
        jnp.zeros((feature.shape[0], 1)),
        jnp.zeros((feature.shape[0], 1)),
    )
    agent.critic = lambda params, key, feature, policy: (
        jnp.array([[0.0], [10.0]]),
        jnp.array([[4.0], [2.0]]),
    )

    loss, _ = SAC._actor_loss(agent, None, None, jnp.ones((2, 1)), None, 0.01)

    np.testing.assert_allclose(loss, -1.0)


def test_other_stochastic_dpg_algorithms_also_use_mode_for_evaluation():
    obs = {"obs": jnp.array([[0.25, -0.5]])}

    tqc = object.__new__(TQC)
    tqc.preproc = lambda params, key, value: value["obs"]
    tqc.actor = lambda params, key, feature: (feature, jnp.full_like(feature, 10.0))

    crossq = object.__new__(CrossQ)
    crossq.preproc = lambda params, key, value: value["obs"]
    crossq.actor = lambda params, key, feature, training: (
        (feature, jnp.full_like(feature, 10.0)),
        {},
    )

    np.testing.assert_allclose(tqc._get_eval_actions(None, obs), jnp.tanh(obs["obs"]))
    np.testing.assert_allclose(crossq._get_eval_actions(None, obs), jnp.tanh(obs["obs"]))


def test_flashsac_entropy_target_and_defaults():
    expected = 0.5 * 4 * np.log(2.0 * np.pi * np.e * 0.15**2)
    np.testing.assert_allclose(entropy_target_from_sigma(4, 0.15), expected)

    params = inspect.signature(SAC.__init__).parameters
    assert params["ent_coef"].default == "auto_0.01"
    assert params["sigma_target"].default == 0.15
    assert params["actor_update_period"].default == 2


def test_sac_actor_and_temperature_update_on_configured_period():
    agent = object.__new__(SAC)
    agent.preproc = lambda params, key, obs: obs["obs"]
    agent.actor = lambda params, key, feature: (
        jnp.full((feature.shape[0], 1), params),
        jnp.full((feature.shape[0], 1), -1.0),
    )
    agent.critic = lambda params, key, feature, actions: (
        params + actions,
        params + 2.0 * actions,
    )
    agent.optimizer = optax.sgd(0.01)
    agent.ent_coef_optimizer = optax.adam(0.01)
    agent.actor_update_period = 2
    agent.target_network_update_tau = 0.01
    agent.target_entropy = 0.0
    agent.auto_entropy = True
    agent.prioritized_replay = False
    agent.scaled_by_reset = False
    agent._gamma = 0.99

    policy_params = jnp.asarray(0.2)
    critic_params = jnp.asarray(1.0)
    target_critic_params = critic_params
    opt_policy_state = agent.optimizer.init(policy_params)
    opt_critic_state = agent.optimizer.init(critic_params)
    log_ent_coef = jnp.log(jnp.asarray(0.01))
    opt_ent_coef_state = agent.ent_coef_optimizer.init(log_ent_coef)
    data = {
        "obses": {"obs": jnp.ones((2, 1))},
        "actions": jnp.zeros((2, 1)),
        "rewards": jnp.zeros((2, 1)),
        "nxtobses": {"obs": jnp.ones((2, 1))},
        "terminateds": jnp.zeros((2, 1)),
    }

    first = SAC._train_step(
        agent,
        policy_params,
        critic_params,
        target_critic_params,
        opt_policy_state,
        opt_critic_state,
        opt_ent_coef_state,
        jax.random.PRNGKey(0),
        1,
        log_ent_coef,
        **data,
    )
    second = SAC._train_step(
        agent,
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        first[5],
        jax.random.PRNGKey(1),
        2,
        first[8],
        **data,
    )

    assert not np.allclose(first[0], policy_params)
    assert int(second[5][0].count) == 1
    np.testing.assert_allclose(second[0], first[0])
    assert int(first[5][0].count) == 1


def test_sac_actor_and_target_use_distinct_fresh_keys():
    agent = object.__new__(SAC)
    agent.optimizer = optax.sgd(0.0)
    agent.actor_update_period = 1
    agent.target_network_update_tau = 0.01
    agent.auto_entropy = False
    agent.prioritized_replay = False
    agent.scaled_by_reset = False
    agent._target = lambda policy, target_critic, rewards, nxtobses, done, key, alpha: (
        jax.random.uniform(key)
    )
    agent._critic_loss = lambda critic, policy, obses, actions, targets, weights, key: (
        targets + 0.0 * critic,
        jnp.zeros((1,)),
    )
    agent._actor_loss = lambda policy, critic, obses, key, alpha: (
        jax.random.uniform(key) + 0.0 * policy,
        jnp.zeros((1, 1)),
    )

    policy_params = jnp.asarray(0.0)
    critic_params = jnp.asarray(0.0)
    log_ent_coef = jnp.log(jnp.asarray(0.01))
    key = jax.random.PRNGKey(7)
    output = SAC._train_step(
        agent,
        policy_params,
        critic_params,
        critic_params,
        agent.optimizer.init(policy_params),
        agent.optimizer.init(critic_params),
        (),
        key,
        1,
        log_ent_coef,
        obses={"obs": jnp.zeros((1, 1))},
        actions=jnp.zeros((1, 1)),
        rewards=jnp.zeros((1, 1)),
        nxtobses={"obs": jnp.zeros((1, 1))},
        terminateds=jnp.zeros((1, 1)),
    )
    target_key, _, actor_key = jax.random.split(key, 3)

    np.testing.assert_allclose(output[6], jax.random.uniform(target_key))
    np.testing.assert_allclose(-output[7], jax.random.uniform(actor_key))
    assert not np.allclose(output[6], -output[7])


def test_sac_cli_uses_flashsac_defaults_without_changing_sibling_algorithms():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    DPG_RUNNER.add_args(parser)
    args = parser.parse_args([])

    assert DPG_RUNNER.algos["SAC"].build(args)["ent_coef"] == "auto_0.01"
    assert DPG_RUNNER.algos["SAC"].build(args)["sigma_target"] == 0.15
    assert DPG_RUNNER.algos["SAC"].build(args)["actor_update_period"] == 2
    assert DPG_RUNNER.algos["TQC"].build(args)["ent_coef"] == "auto"
    assert DPG_RUNNER.algos["CrossQ"].build(args)["ent_coef"] == "auto"
