import haiku as hk
import jax
import jax.numpy as jnp

from model_builder.utils import observation_role_keys


def visual_embedding(mode="normal"):
    if mode != "normal":
        raise ValueError(f"Unknown visual_embedding mode: {mode!r}")
    return hk.Sequential(
        [
            hk.Conv2D(
                32,
                kernel_shape=[8, 8],
                stride=[4, 4],
                padding="VALID",
                w_init=hk.initializers.Orthogonal(scale=1.0),
            ),
            jax.nn.relu,
            hk.Conv2D(
                64,
                kernel_shape=[4, 4],
                stride=[2, 2],
                padding="VALID",
                w_init=hk.initializers.Orthogonal(scale=1.0),
            ),
            jax.nn.relu,
            hk.Conv2D(
                64,
                kernel_shape=[3, 3],
                stride=[1, 1],
                padding="VALID",
                w_init=hk.initializers.Orthogonal(scale=1.0),
            ),
            jax.nn.relu,
            hk.Flatten(),
        ]
    )


class PreProcess(hk.Module):
    @hk.name_like("__call__")
    def __init__(self, state_size, embedding_mode="normal", *, role="actor"):
        super().__init__()
        self.role = role
        self.observation_keys = observation_role_keys(state_size, role)
        self.embedding = {
            key: (visual_embedding(embedding_mode) if len(st) == 3 else lambda x: x)
            for key, st in state_size.items()
            if key in self.observation_keys and (role == "actor" or key.startswith("critic_"))
        }

    def __call__(self, obses: dict[str, jnp.ndarray], shared_features=None) -> jnp.ndarray:
        features = {key: embed(obses[key]) for key, embed in self.embedding.items()}
        if self.role == "critic":
            if shared_features is None:
                raise ValueError("Critic preprocessing requires Actor-owned unified features")
            features.update(shared_features)
        return jnp.concatenate([features[key] for key in self.observation_keys], axis=1)

    def shared_features(self, obses: dict[str, jnp.ndarray]):
        return {
            key: embed(obses[key])
            for key, embed in self.embedding.items()
            if key.startswith("unified_")
        }
