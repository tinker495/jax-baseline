from __future__ import annotations

from model_builder.equinox.dpg.crossq_builder import model_builder_maker as crossq_model_builder_maker


def model_builder_maker(observation_space, action_size, policy_kwargs):
    return crossq_model_builder_maker(observation_space, action_size, policy_kwargs)
