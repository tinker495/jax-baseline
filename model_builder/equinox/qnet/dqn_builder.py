from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.equinox.Module import Dense, NoisyDense, PreProcess, Sequential, sequential_dense
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param


class QNetwork(eqx.Module):
    backbone: Sequential
    value_head: Dense | None
    advantage_head: Dense
    dueling: bool
    noisy: bool

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        dueling: bool,
        noisy: bool,
        node: int,
        hidden_n: int,
        *,
        key: jax.random.KeyArray,
    ) -> None:
        ctor = NoisyDense if noisy else Dense
        key_count = 1 + (2 if dueling else 1)
        keys = jax.random.split(key, key_count)
        backbone, hidden_dim = sequential_dense(
            feature_dim,
            node,
            hidden_n,
            key=keys[0],
            layer_ctor=lambda in_f, out_f, k: ctor(in_f, out_f, key=k),
        )
        self.backbone = backbone
        self.dueling = dueling
        self.noisy = noisy
        head_ctor = lambda in_f, out_f, k: ctor(in_f, out_f, key=k)
        if dueling:
            self.value_head = head_ctor(hidden_dim, 1, keys[1])
            self.advantage_head = head_ctor(hidden_dim, action_dim, keys[2])
        else:
            self.value_head = None
            self.advantage_head = head_ctor(hidden_dim, action_dim, keys[1])

    def __call__(
        self,
        feature: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        hidden = self.backbone(feature, key=key)
        if self.dueling and self.value_head is not None:
            v = self.value_head(hidden, key=key)
            a = self.advantage_head(hidden, key=key)
            return v + a - jnp.max(a, axis=1, keepdims=True)
        return self.advantage_head(hidden, key=key)


class Merged(eqx.Module):
    preproc: PreProcess
    qnet: QNetwork

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def q(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        return self.qnet(feature, key=key)


def model_builder_maker(observation_space, action_space, dueling_model, param_noise, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    embedding_mode = policy_kwargs.pop("embedding_mode", "normal")
    node = policy_kwargs.get("node", 256)
    hidden_n = policy_kwargs.get("hidden_n", 2)

    def model_builder(key=None, print_model=False):
        rng = key if key is not None else jax.random.PRNGKey(0)
        key_pre, key_q = jax.random.split(rng)
        preproc = PreProcess(
            observation_space,
            embedding_mode=embedding_mode,
            key=key_pre,
        )
        feature_dim = preproc.output_size
        qnet = QNetwork(
            feature_dim,
            action_space[0],
            dueling_model,
            param_noise,
            node=node,
            hidden_n=hidden_n,
            key=key_q,
        )
        model = Merged(preproc, qnet)
        params, static = eqx.partition(model, eqx.is_array)
        preproc_fn = get_apply_fn_equinox_module(static, model.preprocess)
        model_fn = get_apply_fn_equinox_module(static, model.q)
        if key is not None:
            if print_model:
                print("------------------build-equinox-model--------------------")
                print_param("", params)
                print("---------------------------------------------------------")
            return preproc_fn, model_fn, params
        return preproc_fn, model_fn

    return model_builder
