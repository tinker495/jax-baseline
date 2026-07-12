from typing import List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


def pop_embedding_mode(policy_kwargs: Optional[dict], default: str = "normal") -> Tuple[dict, str]:
    """Normalize policy_kwargs and split out the embedding_mode entry.

    Returns the (mutated) policy_kwargs dict with ``embedding_mode`` removed and
    the embedding mode string (defaulting to ``default``).
    """
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    embedding_mode = policy_kwargs.pop("embedding_mode", default)
    return policy_kwargs, embedding_mode


def visual_embedding(mode="normal"):
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

    else:
        raise ValueError(f"Unknown visual_embedding mode: {mode!r}")
    return net_fn


class PreProcess(hk.Module):
    def __init__(self, state_size, embedding_mode="normal"):
        super().__init__()
        self.embedding = [
            visual_embedding(embedding_mode) if len(st) == 3 else lambda x: x for st in state_size
        ]

    def __call__(self, obses: List[jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([pre(x) for pre, x in zip(self.embedding, obses)], axis=1)
