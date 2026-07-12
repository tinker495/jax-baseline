from model_builder.flax.dpg.ddpg_builder import _make_model_builder
from model_builder.flax.dpg.simba_ddpg_td3_blocks import Actor, Critic


def model_builder_maker(observation_space, action_size, policy_kwargs):
    return _make_model_builder(
        observation_space,
        action_size,
        policy_kwargs,
        actor_cls=Actor,
        critic_cls=Critic,
        twin_critic=True,
    )
