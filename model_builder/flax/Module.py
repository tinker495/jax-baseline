from typing import Callable, List, Tuple

import flax
import flax.linen as nn
import jax.numpy as jnp


class ResBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
        )(inputs)
        x = nn.GroupNorm(num_groups=max(self.filters // 32, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
        )(x)
        x = nn.GroupNorm(num_groups=max(self.filters // 32, 1))(x)
        x = nn.relu(x)
        return x + inputs


class ImpalaBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            self.filters,
            kernel_size=3,
            strides=1,
            padding="SAME",
            kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResBlock(self.filters)(x)
        x = ResBlock(self.filters)(x)
        return x


def flatten_fn(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape((x.shape[0], -1))


def visual_embedding(mode: str = "normal", flatten=True) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if mode == "resnet":
        mul = 1
        net = nn.Sequential(
            [
                ImpalaBlock(16 * mul),
                ImpalaBlock(32 * mul),
                ImpalaBlock(32 * mul),
                flatten_fn if flatten else lambda x: x,
            ]
        )

    elif mode == "normal":
        net = nn.Sequential(
            [
                nn.Conv(
                    32,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    64,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    64,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten_fn if flatten else lambda x: x,
            ]
        )

    elif mode == "simple":

        net = nn.Sequential(
            [
                nn.Conv(
                    16,
                    kernel_size=[8, 8],
                    strides=[4, 4],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    32,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten_fn if flatten else lambda x: x,
            ]
        )

    elif mode == "minimum":
        net = nn.Sequential(
            [
                nn.Conv(
                    16,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                nn.Conv(
                    32,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    padding="VALID",
                    kernel_init=flax.linen.initializers.orthogonal(scale=1.0),
                ),
                nn.relu,
                flatten_fn if flatten else lambda x: x,
            ]
        )
    elif mode == "none":
        net = flatten
    return net


class PreProcess(nn.Module):
    states_size: List[Tuple[int, ...]]
    embedding_mode: str = "normal"
    flatten: bool = True

    def setup(self):
        self.embedding = [
            visual_embedding(self.embedding_mode, self.flatten) if len(st) == 3 else lambda x: x
            for st in self.states_size
        ]

    @nn.compact
    def __call__(self, states: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre, x in zip(self.embedding, states)], axis=1)

    @property
    def output_size(self):
        return sum(
            [
                pre(jnp.zeros((1,) + st)).shape[1]
                for pre, st in zip(self.embedding, self.states_size)
            ]
        )
