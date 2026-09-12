import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_factorized_uniform
from model_builder.flax.layers import (
    LOG_STD_MEAN,
    LOG_STD_SCALE,
    Dense,
    SimbaV2Head,
    network_body,
)
from model_builder.flax.Module import BatchReNorm, PreProcess
from model_builder.model_config import (
    ACTIVATIONS,
    LayerConfig,
    MLPConfig,
    ModelConfig,
    ResidualConfig,
)
from model_builder.utils import (
    dummy_observation,
    get_critic_apply_fn,
    print_flax_model_summary,
    split_actor_critic_kwargs,
)


class Actor(nn.Module):
    action_size: tuple
    network: ModelConfig = MLPConfig()
    layer: type[nn.Dense] = Dense

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> tuple[jax.Array, jax.Array]:
        feature = network_body(features, self.network, self.layer)
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            mu = SimbaV2Head(
                self.network.blocks[-1], self.action_size[0], kernel_init=clip_factorized_uniform(3)
            )(feature)
            log_std = SimbaV2Head(
                self.network.blocks[-1],
                self.action_size[0],
                use_bias=True,
                kernel_init=clip_factorized_uniform(3),
                bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
            )(feature)
        else:
            mu = self.layer(self.action_size[0], kernel_init=clip_factorized_uniform(3))(feature)
            log_std = self.layer(
                self.action_size[0],
                kernel_init=clip_factorized_uniform(3),
                bias_init=lambda key, shape, dtype: jnp.full(shape, 10.0, dtype=dtype),
            )(feature)
        return mu, LOG_STD_MEAN + LOG_STD_SCALE * jax.nn.tanh(log_std / LOG_STD_SCALE)


class Critic(nn.Module):
    network: ModelConfig = MLPConfig((LayerConfig(2048, "tanh"),) * 2)
    layer: type[nn.Dense] = Dense

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, actions: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        concat = jnp.concatenate([features, actions], axis=1)
        if isinstance(self.network, ResidualConfig) and self.network.kind == "simbav2":
            return SimbaV2Head(self.network.blocks[-1], 1)(network_body(concat, self.network))
        feature = BatchReNorm(use_running_average=not training)(concat)
        if isinstance(self.network, MLPConfig):
            for layer in self.network.layers:
                feature = self.layer(layer.units)(feature)
                feature = BatchReNorm(use_running_average=not training)(feature)
                feature = ACTIVATIONS[layer.activation](feature)
        else:
            feature = network_body(feature, self.network, self.layer)
        return self.layer(1, kernel_init=clip_factorized_uniform(3))(feature)


def model_builder_maker(observation_space, action_size, policy_kwargs):
    actor_kwargs, critic_kwargs = split_actor_critic_kwargs(
        policy_kwargs,
        critic_default=MLPConfig((LayerConfig(2048, "tanh"),) * 2),
        allowed_types=("mlp", "simba", "simbav2"),
    )

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=actor_kwargs["network"].embedding_mode,
                    role="actor",
                )
                self.act = Actor(action_size, **actor_kwargs)

            def __call__(self, observation):
                return self.act(self.preproc(observation))

            def shared_features(self, observation):
                return self.preproc.shared_features(observation)

        class Merged_Critic(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space,
                    embedding_mode=critic_kwargs["network"].embedding_mode,
                    role="critic",
                )
                self.crit1 = Critic(**critic_kwargs)
                self.crit2 = Critic(**critic_kwargs)

            def __call__(self, observation, shared_features, action, training: bool = True):
                feature = self.preproc(observation, shared_features)
                return self.crit1(feature, action, training), self.crit2(feature, action, training)

        actor_model = Merged_Actor()
        critic_model = Merged_Critic()
        actor_fn = get_apply_fn_flax_module(actor_model)
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
        action = np.zeros((1, *action_size), dtype=np.float32)
        actor_key, critic_key = jax.random.split(key)
        policy_params = actor_model.init(actor_key, observation)
        shared_features = shared_preproc_fn(policy_params, None, observation)
        critic_params = critic_model.init(critic_key, observation, shared_features, action, True)
        print_flax_model_summary(
            print_model,
            key,
            (actor_model, observation),
            (critic_model, observation, shared_features, action, True),
        )
        return actor_fn, critic_fn, policy_params, critic_params

    return model_builder
