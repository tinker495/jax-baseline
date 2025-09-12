from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.module import (  # pylint: disable=g-multiple-import
    Module,
    compact,
    merge_param,
)
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax import lax
from jax.nn import initializers


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


def visual_embedding(
    mode: str = "normal", flatten=True, **kwargs
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if mode == "resnet":
        multiple = kwargs.get("multiple", 1)
        net = nn.Sequential(
            [
                ImpalaBlock(16 * multiple),
                ImpalaBlock(32 * multiple),
                ImpalaBlock(32 * multiple),
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
    pre_postprocess: Callable = lambda x: x  # Identity function
    multiple: int = 1

    def setup(self):
        self.embedding = [
            visual_embedding(self.embedding_mode, self.flatten, multiple=self.multiple)
            if len(st) == 3
            else lambda x: x
            for st in self.states_size
        ]

    @nn.compact
    def __call__(self, obses: List[jnp.ndarray]) -> jnp.ndarray:
        return self.pre_postprocess(
            jnp.concatenate([pre(x) for pre, x in zip(self.embedding, obses)], axis=1)
        )

    @property
    def output_size(self):
        return sum(
            [
                pre(jnp.zeros((1,) + st)).shape[1]
                for pre, st in zip(self.embedding, self.states_size)
            ]
        )


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class BatchReNorm(Module):
    """BatchReNorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation:
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    r_max: float = 3.0
    d_max: float = 5.0
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jnp.clip(std / ra_std, 1 / self.r_max, self.r_max)
                r = lax.stop_gradient(r)
                d = jnp.clip((mean - ra_mean.value) / ra_std, -self.d_max, self.d_max)
                d = lax.stop_gradient(d)
                custom_mean = mean - (std * d / r)
                custom_var = (var + self.epsilon) / (r * r) - self.epsilon

                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
