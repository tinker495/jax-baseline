from model_builder.flax.qnet.spr_modules import make_spr_style_builder_maker


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
        embedding_default="resnet",
        preproc_multiple=4,
    )
