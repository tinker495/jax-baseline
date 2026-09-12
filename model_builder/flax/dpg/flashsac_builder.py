from collections.abc import Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_baselines.math.param_updates import project_unit_norm_params
from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.Module import PreProcess
from model_builder.model_config import ACTIVATIONS, ResidualConfig
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class UnitBatchNorm(nn.Module):
    @nn.compact
    def __call__(self, inputs: jax.Array, training: bool) -> jax.Array:
        scale = self.param("scale", nn.initializers.ones_init(), (inputs.shape[-1],))
        bias = self.param("bias", nn.initializers.zeros_init(), (inputs.shape[-1],))
        running_mean = self.variable("batch_stats", "mean", jnp.zeros, (inputs.shape[-1],))
        running_var = self.variable("batch_stats", "var", jnp.ones, (inputs.shape[-1],))
        if training:
            if inputs.shape[0] < 2:
                raise ValueError("FlashSAC training BatchNorm requires at least two samples")
            mean = jnp.mean(inputs, axis=0)
            variance = jnp.mean(jnp.square(inputs - mean), axis=0)
            # PyTorch normalizes with population variance but stores an unbiased estimate.
            running_mean.value = 0.99 * running_mean.value + 0.01 * jax.lax.stop_gradient(mean)
            running_var.value = 0.99 * running_var.value + 0.01 * jax.lax.stop_gradient(
                variance * (inputs.shape[0] / (inputs.shape[0] - 1))
            )
        else:
            mean, variance = running_mean.value, running_var.value
        return (inputs - mean) * jax.lax.rsqrt(variance + 1e-5) * scale + bias


class Encoder(nn.Module):
    network: ResidualConfig

    @nn.compact
    def __call__(self, feature: jax.Array, training: bool) -> jax.Array:
        feature = UnitBatchNorm(name="input_norm")(feature, training)
        feature = nn.Dense(
            self.network.blocks[0],
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(),
            name="embed",
        )(feature)
        for block, width in enumerate(self.network.blocks):
            if feature.shape[-1] != width:
                feature = nn.Dense(
                    width,
                    use_bias=False,
                    kernel_init=nn.initializers.orthogonal(),
                    name=f"block_{block}_project",
                )(feature)
            residual = feature
            feature = nn.Dense(
                width * 4,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"block_{block}_expand",
            )(feature)
            feature = UnitBatchNorm(name=f"block_{block}_norm1")(feature, training)
            feature = ACTIVATIONS[self.network.activation](feature)
            feature = nn.Dense(
                width,
                use_bias=False,
                kernel_init=nn.initializers.orthogonal(),
                name=f"block_{block}_contract",
            )(feature)
            feature = UnitBatchNorm(name=f"block_{block}_norm2")(feature, training)
            feature = ACTIVATIONS[self.network.activation](feature) + residual
        return nn.RMSNorm(epsilon=1e-6, name="post_norm")(feature)


class Actor(nn.Module):
    action_dim: int
    network: ResidualConfig = ResidualConfig("flashsac", (128, 128))

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        feature = Encoder(self.network)(features, training)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(), name="mean")(
            feature
        )
        raw_log_std = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(), name="std"
        )(feature)
        return mean, -10.0 + 6.0 * (1.0 + jax.nn.tanh(raw_log_std))


class Critic(nn.Module):
    network: ResidualConfig = ResidualConfig("flashsac")
    n_atoms: int = 101

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, actions: jax.Array, training: bool = False
    ) -> jax.Array:
        feature = Encoder(self.network)(jnp.concatenate((features, actions), axis=-1), training)
        return nn.Dense(self.n_atoms, kernel_init=nn.initializers.orthogonal(), name="value")(
            feature
        )


def model_builder_maker(
    observation_space: Mapping[str, Sequence[int]],
    action_size: Sequence[int],
    policy_kwargs,
):
    """Build FlashSAC's residual policy and independent categorical twin critics."""
    model_options = {} if policy_kwargs is None else dict(policy_kwargs)
    n_atoms = model_options.pop("n_atoms", 101)
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        model_options,
        actor_default=ResidualConfig("flashsac", (128, 128)),
        critic_default=ResidualConfig("flashsac"),
        allowed_embeddings=("normal",),
    )
    if policy_kwargs is not None:
        policy_kwargs["actor_model"] = actor_kwargs["network"]
        policy_kwargs["critic_model"] = critic_kwargs["network"]
    unsupported = set(actor_kwargs) - {"network"}
    if unsupported:
        raise ValueError(f"Unsupported FlashSAC policy options: {sorted(unsupported)}")
    if type(n_atoms) is not int or n_atoms < 2:
        raise ValueError("FlashSAC n_atoms must be an integer >= 2")
    if not observation_space or any(
        len(shape) != 1 or shape[0] < 1 for shape in observation_space.values()
    ):
        raise ValueError("FlashSAC requires nonempty vector observations")
    if len(action_size) != 1 or action_size[0] < 1:
        raise ValueError("FlashSAC requires a nonempty vector action space")
    observation_space = {name: tuple(shape) for name, shape in observation_space.items()}

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=actor_kwargs["network"].embedding_mode,
                    role="actor",
                )
                self.act = Actor(action_size[0], **actor_kwargs)

            def __call__(self, observation, training: bool = False):
                return self.act(self.preproc(observation), training)

            def shared_features(self, observation):
                return self.preproc.shared_features(observation)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=critic_kwargs["network"].embedding_mode,
                    role="critic",
                )
                self.crit1 = Critic(n_atoms=n_atoms, **critic_kwargs)
                self.crit2 = Critic(n_atoms=n_atoms, **critic_kwargs)

            def __call__(self, observation, shared_features, actions, training: bool = False):
                feature = self.preproc(observation, shared_features)
                return self.crit1(feature, actions, training), self.crit2(
                    feature, actions, training
                )

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model, mutable=["batch_stats"])
        shared_preproc_fn = get_apply_fn_flax_module(
            actor_model, method=actor_model.shared_features
        )
        critic_fn = get_critic_apply_fn(
            get_apply_fn_flax_module(critic_model, mutable=["batch_stats"]),
            shared_preproc_fn,
        )
        if key is None:
            return actor_fn, critic_fn
        observation = dummy_observation(observation_space)
        action = jnp.zeros((1, *action_size), dtype=jnp.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_variables = actor_model.init(actor_key, observation, False)
        policy_variables["params"] = project_unit_norm_params(policy_variables["params"])
        shared_features = shared_preproc_fn(policy_variables, None, observation)
        critic_variables = critic_model.init(
            critic_key, observation, shared_features, action, False
        )
        critic_variables["params"] = project_unit_norm_params(critic_variables["params"])
        print_flax_model_summary(
            print_model,
            key,
            (actor_model, observation, False),
            (critic_model, observation, shared_features, action, False),
        )
        return actor_fn, critic_fn, policy_variables, critic_variables

    return model_builder
