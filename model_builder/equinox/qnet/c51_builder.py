from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from model_builder.equinox.Module import Dense, NoisyDense, PreProcess, sequential_dense
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param


def _layer_ctor(noisy: bool):
    return NoisyDense if noisy else Dense


def _init_layer(ctor, in_dim, out_dim, key, **kwargs):
    if ctor is NoisyDense:
        kwargs.pop("use_bias", None)
    return ctor(in_dim, out_dim, key=key, **kwargs)


class C51Network(eqx.Module):
    backbone: eqx.Module
    value_head: Dense | None
    advantage_head: Dense
    dueling: bool
    action_dim: int
    support_n: int

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        support_n: int,
        *,
        node: int,
        hidden_n: int,
        dueling: bool,
        noisy: bool,
        key: jax.random.KeyArray,
    ) -> None:
        ctor = _layer_ctor(noisy)
        split_count = 3 if dueling else 2
        keys = jax.random.split(key, split_count)
        key_backbone = keys[0]
        backbone, hidden_dim = sequential_dense(
            feature_dim,
            node,
            hidden_n,
            key=key_backbone,
            layer_ctor=lambda in_f, out_f, k: ctor(in_f, out_f, key=k),
        )
        self.backbone = backbone
        self.dueling = dueling
        self.action_dim = action_dim
        self.support_n = support_n
        if dueling:
            key_v, key_a = keys[1], keys[2]
            self.value_head = _init_layer(ctor, hidden_dim, support_n, key_v)
            self.advantage_head = _init_layer(
                ctor, hidden_dim, action_dim * support_n, key_a
            )
        else:
            self.value_head = None
            key_a = keys[1]
            self.advantage_head = _init_layer(
                ctor, hidden_dim, action_dim * support_n, key_a
            )

    def __call__(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        backbone_key = None
        value_key = None
        adv_key = None
        if key is not None:
            key, backbone_key = jax.random.split(key)
            if self.dueling:
                backbone_key, value_key, adv_key = jax.random.split(backbone_key, 3)
            else:
                backbone_key, adv_key = jax.random.split(backbone_key)
        hidden = self.backbone(feature, key=backbone_key)
        if not self.dueling:
            logits = self.advantage_head(hidden, key=adv_key)
            logits = logits.reshape((hidden.shape[0], self.action_dim, self.support_n))
            return jax.nn.softmax(logits, axis=-1)
        value = self.value_head(hidden, key=value_key)
        value = value.reshape((hidden.shape[0], 1, self.support_n))
        advantage = self.advantage_head(hidden, key=adv_key)
        advantage = advantage.reshape((hidden.shape[0], self.action_dim, self.support_n))
        logits = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return jax.nn.softmax(logits, axis=-1)


class Merged(eqx.Module):
    preproc: PreProcess
    qnet: C51Network

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def q(self, feature: jnp.ndarray, *, key: jax.random.KeyArray | None = None) -> jnp.ndarray:
        return self.qnet(feature, key=key)


def model_builder_maker(
    observation_space,
    action_space,
    dueling_model,
    param_noise,
    categorial_bar_n,
    policy_kwargs,
):
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
        qnet = C51Network(
            feature_dim,
            action_space[0],
            support_n=categorial_bar_n,
            node=node,
            hidden_n=hidden_n,
            dueling=dueling_model,
            noisy=param_noise,
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
