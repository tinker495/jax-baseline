from model_builder.flax.qnet.spr_modules import make_spr_style_builder_maker
from model_builder.model_config import MLPConfig


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    return make_spr_style_builder_maker(
        observation_space,
        action_space,
        dueling_model,
        param_noise,
        categorial_bar_n,
        policy_kwargs,
        model_default=MLPConfig(embedding_mode="resnet"),
        preproc_multiple=4,
    )
