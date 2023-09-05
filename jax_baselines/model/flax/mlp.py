import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from typing import Any, Callable, Dict, Optional, Tuple, Union, List


class NoisyDense(nn.Dense):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension with noisy params.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
            inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        else:
            dot_general = self.dot_general
            y = dot_general(
                inputs,
                kernel,
                (((inputs.ndim - 1,), (0,)), ((), ())),
                precision=self.precision,
            )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MLP(nn.Module):
    @flax.struct.dataclass
    class HyperParams:
        hidden_layers: List[int]
        layer_type: nn.Module = nn.Dense
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
        kernel_init: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
        ] = nn.initializers.xavier_uniform()
        bias_init: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
        ] = nn.initializers.normal(stddev=1e-6)

    hyperparams: HyperParams

    def setup(self):
        layer = self.hyperparams.layer_type.partial(
            kernel_init=self.hyperparams.kernel_init,
            bias_init=self.hyperparams.bias_init,
        )
        self.layers = [layer(features) for features in self.hyperparams.hidden_layers]
        self.activation = self.hyperparams.activation

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x
