from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from model_builder.equinox.Module import Dense, NoisyDense, PreProcess
from model_builder.equinox.apply import get_apply_fn_equinox_module
from model_builder.utils import print_param

_MAX_BASIS = 128


def _layer_ctor(noisy: bool):
    return NoisyDense if noisy else Dense


def _init_layer(ctor, in_dim, out_dim, key, **kwargs):
    if ctor is NoisyDense:
        kwargs.pop("use_bias", None)
    return ctor(in_dim, out_dim, key=key, **kwargs)


class IQN(eqx.Module):
    feature_proj: Dense
    quantile_proj: Dense
    backbone: tuple
    dueling: bool
    action_dim: int
    noisy: bool
    value_head: Dense | None
    advantage_head: Dense
    pi_mtx: jnp.ndarray

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        node: int,
        hidden_n: int,
        dueling: bool,
        noisy: bool,
        key: jax.random.KeyArray,
    ) -> None:
        ctor = _layer_ctor(noisy)
        keys = jax.random.split(key, hidden_n + 4 if dueling else hidden_n + 3)
        self.feature_proj = Dense(feature_dim, node, key=keys[0])
        self.quantile_proj = Dense(_MAX_BASIS, node, key=keys[1])
        layers = []
        in_dim = node
        for i in range(hidden_n):
            layer_key = keys[2 + i]
            layers.append(_init_layer(ctor, in_dim, node, layer_key))
            layers.append(lambda x: jax.nn.relu(x))
        self.backbone = tuple(layers)
        head_index = 2 + hidden_n
        if dueling:
            self.value_head = _init_layer(ctor, node, 1, keys[head_index])
            self.advantage_head = _init_layer(ctor, node, action_dim, keys[head_index + 1])
        else:
            self.value_head = None
            self.advantage_head = _init_layer(ctor, node, action_dim, keys[head_index])
        self.dueling = dueling
        self.action_dim = action_dim
        self.noisy = noisy
        self.pi_mtx = jnp.pi * (jnp.arange(_MAX_BASIS, dtype=jnp.float32) + 1.0)

    def _apply_backbone(self, x: jnp.ndarray, *, key: jax.random.KeyArray | None) -> jnp.ndarray:
        current = x
        layer_key = key
        for layer in self.backbone:
            if isinstance(layer, eqx.Module):
                subkey = None
                if layer_key is not None:
                    layer_key, subkey = jax.random.split(layer_key)
                current = layer(current, key=subkey)
            else:
                current = layer(current)
        return current

    def __call__(
        self,
        feature: jnp.ndarray,
        tau: jnp.ndarray,
        *,
        key: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        batch_size, feature_dim = feature.shape
        num_tau = tau.shape[1]

        projected_feature = jax.nn.relu(self.feature_proj(feature))

        cos_basis = jnp.cos(jnp.expand_dims(tau, -1) * self.pi_mtx)
        quantile_embedding = jax.nn.relu(self.quantile_proj(cos_basis))
        mul_embedding = projected_feature[:, None, :] * quantile_embedding
        flat_embedding = mul_embedding.reshape(batch_size * num_tau, -1)

        backbone_key = None
        value_key = None
        adv_key = None
        if key is not None:
            if self.dueling:
                key, backbone_key = jax.random.split(key)
                backbone_key, value_key, adv_key = jax.random.split(backbone_key, 3)
            else:
                backbone_key, adv_key = jax.random.split(key)

        hidden = self._apply_backbone(flat_embedding, key=backbone_key)

        if self.dueling:
            value = self.value_head(hidden, key=value_key)
            advantage = self.advantage_head(hidden, key=adv_key)
            value = value.reshape(batch_size, num_tau, 1)
            advantage = advantage.reshape(batch_size, num_tau, self.action_dim)
            q = value + advantage - jnp.max(advantage, axis=2, keepdims=True)
        else:
            advantage = self.advantage_head(hidden, key=adv_key)
            q = advantage.reshape(batch_size, num_tau, self.action_dim)
        return jnp.transpose(q, (0, 2, 1))


class Merged(eqx.Module):
    preproc: PreProcess
    qnet: IQN

    def preprocess(self, obses, *, key: jax.random.KeyArray | None = None):
        obses = [jnp.asarray(o) for o in obses]
        return self.preproc(obses, key=key)

    def q(self, feature: jnp.ndarray, tau: jnp.ndarray, *, key: jax.random.KeyArray | None = None):
        return self.qnet(feature, tau, key=key)


def model_builder_maker(
    observation_space,
    action_space,
    dueling_model,
    param_noise,
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
        qnet = IQN(
            feature_dim,
            action_space[0],
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
