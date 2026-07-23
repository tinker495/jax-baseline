import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.math.param_updates import project_dense_kernels
from jax_baselines.XQC.xqc import XQC
from model_builder.flax.dpg.xqc_builder import model_builder_maker


def test_project_dense_kernels_normalizes_only_dense_weights():
    params = {
        "params": {
            "Dense_0": {
                "kernel": jnp.asarray([[3.0, 0.0], [4.0, 2.0]]),
                "bias": jnp.ones(2),
            },
            "BatchNorm_0": {"scale": jnp.asarray([3.0, 4.0])},
        }
    }

    projected = project_dense_kernels(params)

    np.testing.assert_allclose(np.linalg.norm(projected["params"]["Dense_0"]["kernel"], axis=0), 1)
    np.testing.assert_array_equal(projected["params"]["Dense_0"]["bias"], np.ones(2))
    np.testing.assert_array_equal(projected["params"]["BatchNorm_0"]["scale"], [3, 4])


def test_reward_normalization_constructor_defaults_are_xqc_only():
    xqc_default = inspect.signature(XQC.__init__).parameters["reward_normalization"].default
    dpg_default = (
        inspect.signature(Deteministic_Policy_Gradient_Family.__init__)
        .parameters["reward_normalization"]
        .default
    )
    qnet_default = (
        inspect.signature(Q_Network_Family.__init__).parameters["reward_normalization"].default
    )

    assert xqc_default is True
    assert dpg_default is False
    assert qnet_default is False


def test_xqc_categorical_q_uses_fixed_support_expectation():
    agent = XQC.__new__(XQC)
    agent.value_support = jnp.asarray([-5.0, 0.0, 5.0])

    logits = jnp.log(jnp.asarray([[0.2, 0.3, 0.5]]))

    np.testing.assert_allclose(agent._categorical_q(logits), [1.5], rtol=1e-6)


def test_xqc_categorical_train_step_updates_online_and_target_critics():
    agent = XQC.__new__(XQC)
    builder = model_builder_maker({"obs": [4]}, [2], {"node": 16, "embedding_mode": "normal"})
    (
        agent.preproc,
        agent.actor,
        agent.critic,
        policy_params,
        critic_params,
    ) = builder(jax.random.PRNGKey(0))
    target_critic_params = critic_params
    agent.optimizer = optax.adam(3e-4)
    agent.ent_coef_optimizer = optax.adam(3e-4)
    agent.auto_entropy = True
    agent.target_entropy = 1.0
    agent.policy_delay = 3
    agent.target_network_update_tau = 0.005
    agent.prioritized_replay = False
    agent.scaled_by_reset = False
    agent._gamma = 0.99
    agent.value_min = -5.0
    agent.value_max = 5.0
    agent.n_atoms = 101
    agent.value_support = jnp.linspace(-5.0, 5.0, 101)
    agent.support_delta = 0.1
    log_ent_coef = jnp.log(0.01)

    result = jax.jit(agent._train_step)(
        policy_params,
        critic_params,
        target_critic_params,
        agent.optimizer.init(policy_params),
        agent.optimizer.init(critic_params),
        agent.ent_coef_optimizer.init(log_ent_coef),
        jax.random.PRNGKey(1),
        1,
        log_ent_coef,
        {"obs": jnp.zeros((4, 4))},
        jnp.zeros((4, 2)),
        jnp.ones((4, 1)),
        {"obs": jnp.ones((4, 4))},
        jnp.zeros((4, 1)),
    )

    assert np.isfinite(result[6])
    assert np.isfinite(result[7])
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(target_critic_params),
            jax.tree_util.tree_leaves(result[2]),
        )
    )
