from model_builder.haiku.dpg.ddpg_builder import _make_model_builder


def model_builder_maker(observation_space, action_size, policy_kwargs):
    return _make_model_builder(
        observation_space,
        action_size,
        policy_kwargs,
        twin_critic=True,
    )
