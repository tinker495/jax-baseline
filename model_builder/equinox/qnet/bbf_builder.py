from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import PreProcess
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.equinox.qnet.c51_builder import C51Network
from model_builder.equinox.qnet.spr_builder import Transition, Projection, Prediction, SPR
from model_builder.utils import print_param


def model_builder_maker(
    observation_space,
    action_space,
    dueling_model,
    param_noise,
    categorial_bar_n,
    policy_kwargs,
):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_q, key_trans, key_proj, key_pred = jax.random.split(rng, 5)
        preproc = PreProcess(
            observation_space,
            embedding_mode="resnet",
            key=key_pre,
            flatten=False,
            multiple=4,
        )
        dummy_obs = [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
        feature_sample = preproc(dummy_obs)
        flat_dim = int(np.prod(feature_sample.shape[1:]))
        qnet = C51Network(
            flat_dim,
            action_space[0],
            categorial_bar_n,
            node=node,
            hidden_n=hidden_n,
            dueling=dueling_model,
            noisy=param_noise,
            key=key_q,
        )
        transition = Transition(feature_sample.shape[1:], action_space[0], key=key_trans)
        projection = Projection(feature_sample.shape[1:], key=key_proj)
        prediction = Prediction(projection.embed_size, key=key_pred)
        model = SPR(preproc, qnet, transition, projection, prediction)
        params, static = eqx.partition(model, eqx.is_array)
        preproc_fn = get_apply_fn_equinox_module(static, model.preprocess)
        q_fn = get_apply_fn_equinox_module(static, model.q)
        transition_fn = get_apply_fn_equinox_module(static, model.transition_forward)
        projection_fn = get_apply_fn_equinox_module(static, model.projection_forward)
        prediction_fn = get_apply_fn_equinox_module(static, model.prediction_forward)
        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("bbf", params)
                print("---------------------------------------------------------")
            return preproc_fn, q_fn, transition_fn, projection_fn, prediction_fn, params
        return preproc_fn, q_fn, transition_fn, projection_fn, prediction_fn

    return model_builder
