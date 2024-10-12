from typing import List

import haiku as hk
import jax
import jax.numpy as jnp


def visual_embedding(mode="simple"):
    if mode == "normal":

        def net_fn(x):
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
            )(x)

    elif mode == "simple":

        def net_fn(x):
            return hk.Sequential(
                [
                    hk.Conv2D(
                        16,
                        kernel_shape=[8, 8],
                        stride=[4, 4],
                        padding="VALID",
                        w_init=hk.initializers.Orthogonal(scale=1.0),
                    ),
                    jax.nn.relu,
                    hk.Conv2D(
                        32,
                        kernel_shape=[4, 4],
                        stride=[2, 2],
                        padding="VALID",
                        w_init=hk.initializers.Orthogonal(scale=1.0),
                    ),
                    jax.nn.relu,
                    hk.Flatten(),
                ]
            )(x)

    elif mode == "minimum":

        def net_fn(x):
            return hk.Sequential(
                [
                    hk.Conv2D(16, kernel_shape=[3, 3], stride=[1, 1], padding="VALID"),
                    jax.nn.relu,
                    hk.Conv2D(32, kernel_shape=[4, 4], stride=[2, 2], padding="VALID"),
                    jax.nn.relu,
                    hk.Flatten(),
                ]
            )(x)

    elif mode == "none":
        net_fn = hk.Flatten()
    return net_fn


class PreProcess(hk.Module):
    def __init__(self, state_size, embedding_mode="normal"):
        super().__init__()
        self.embedding = [
            visual_embedding(embedding_mode) if len(st) == 3 else lambda x: x for st in state_size
        ]

    def __call__(self, obses: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre, x in zip(self.embedding, obses)], axis=1)
