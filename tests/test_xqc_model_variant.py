import jax
import jax.numpy as jnp

from model_builder.flax.dpg.xqc_builder import model_builder_maker


def test_xqc_builder_uses_four_layer_batchnorm_mlp():
    builder = model_builder_maker(
        {"obs": [4]},
        [2],
        {"node": 16, "hidden_n": 1, "embedding_mode": "normal"},
    )
    preproc, actor, critic, policy_params, critic_params = builder(jax.random.PRNGKey(0))

    actor_params = policy_params["params"]["act"]
    assert set(policy_params["batch_stats"]["act"]) == {f"BatchNorm_{index}" for index in range(5)}
    assert actor_params["Dense_0"]["kernel"].shape == (4, 16)
    assert "bias" not in actor_params["Dense_0"]
    assert actor_params["Dense_3"]["kernel"].shape == (16, 16)

    critic_one = critic_params["params"]["crit1"]
    assert set(critic_params["batch_stats"]["crit1"]) == {
        f"BatchNorm_{index}" for index in range(5)
    }
    assert critic_one["Dense_0"]["kernel"].shape == (6, 32)
    assert "bias" not in critic_one["Dense_0"]
    assert critic_one["Dense_3"]["kernel"].shape == (32, 32)
    assert critic_one["Dense_4"]["kernel"].shape == (32, 101)

    feature = preproc(policy_params, None, {"obs": jnp.zeros((2, 4), dtype=jnp.float32)})
    (mu, log_std), actor_updates = actor(policy_params, None, feature, True)
    (q1, q2), critic_updates = critic(
        critic_params, None, feature, jnp.zeros((2, 2), dtype=jnp.float32), True
    )
    assert mu.shape == log_std.shape == (2, 2)
    assert q1.shape == q2.shape == (2, 101)
    assert set(actor_updates) == set(critic_updates) == {"batch_stats"}
