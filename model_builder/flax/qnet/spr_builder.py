import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_uniform_initializers
from model_builder.flax.layers import Dense, NoisyDense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param


class Projection(nn.Module):
    embed_size: int = 128
    node: int = 512
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        feature = jnp.reshape(feature, (feature.shape[0], -1))
        projection = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        projection = self.layer(self.embed_size)(projection)
        return projection


class Transition(nn.Module):
    node: int = 64
    hidden_n: int = 2
    layer: callable = lambda ch: nn.Sequential(
        [
            nn.Conv(
                ch,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                kernel_init=nn.initializers.orthogonal(scale=1.0),
            ),
            nn.GroupNorm(num_groups=4),
            nn.relu,
        ]
    )
    last: callable = lambda ch: nn.Conv(
        ch,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="SAME",
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    )

    @nn.compact
    def __call__(self, feature: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, action], axis=-1)
        x = nn.Sequential([self.layer(self.node) for i in range(self.hidden_n)])(concat)
        feature = self.last(feature.shape[-1])(x)
        return feature


class Prediction(nn.Module):
    node: int = 128
    hidden_n: int = 1
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        x = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        feature = self.layer(feature.shape[-1])(x)
        return feature


class Model(nn.Module):
    action_size: int
    node: int
    hidden_n: int
    noisy: bool
    dueling: bool
    categorial_bar_n: int

    def setup(self) -> None:
        if not self.noisy:
            self.layer = Dense
        else:
            self.layer = NoisyDense

    @nn.compact
    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        if not self.dueling:
            q_net = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.categorial_bar_n,
                        kernel_init=clip_uniform_initializers(-0.03, 0.03),
                    ),
                    lambda x: jnp.reshape(x, (-1, self.action_size[0], self.categorial_bar_n)),
                ]
            )(feature)
            return jax.nn.softmax(q_net, axis=2)
        else:
            v = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.categorial_bar_n, kernel_init=clip_uniform_initializers(-0.03, 0.03)
                    ),
                    lambda x: jnp.reshape(x, (-1, 1, self.categorial_bar_n)),
                ]
            )(feature)
            a = nn.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.categorial_bar_n,
                        kernel_init=clip_uniform_initializers(-0.03, 0.03),
                    ),
                    lambda x: jnp.reshape(x, (-1, self.action_size[0], self.categorial_bar_n)),
                ]
            )(feature)
            q = v + a - jnp.mean(a, axis=1, keepdims=True)
            return jax.nn.softmax(q, axis=-1)


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Merged(nn.Module):
            def setup(self):
                self.preproc = PreProcess(
                    observation_space, embedding_mode=embedding_mode, flatten=False
                )
                self.qnet = Model(
                    action_space,
                    dueling=dueling_model,
                    noisy=param_noise,
                    categorial_bar_n=categorial_bar_n,
                    **policy_kwargs
                )
                self.tran = Transition()
                self.proj = Projection()
                self.pred = Prediction()

            def __call__(self, x, action):
                feature = self.preprocess(x)
                q = self.q(feature)
                transition = self.transition(feature, action)
                projection = self.projection(transition)
                prediction = self.prediction(projection)
                return q, transition, projection, prediction

            def preprocess(self, x):
                return self.preproc(x)

            def q(self, x):
                x = jnp.reshape(x, (x.shape[0], -1))
                return self.qnet(x)

            def transition(self, x, action):
                batch_size = x.shape[0]
                tile_shape = (1, x.shape[1], x.shape[2], 1)
                action = jnp.reshape(action, (batch_size, 1, 1, -1))
                action = jnp.tile(action, tile_shape)
                return self.tran(x, action)

            def projection(self, x):
                return self.proj(x)

            def prediction(self, x):
                return self.pred(x)

        model = Merged()
        preproc_fn = get_apply_fn_flax_module(model, model.preprocess)
        model_fn = get_apply_fn_flax_module(model, model.q)
        transition_fn = get_apply_fn_flax_module(model, model.transition)
        projection_fn = get_apply_fn_flax_module(model, model.projection)
        prediction_fn = get_apply_fn_flax_module(model, model.prediction)

        if key is not None:
            params = model.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
                jnp.zeros((1, action_space[0]), dtype=np.float32),
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", params)
                print("------------------------------------------------------")
            return preproc_fn, model_fn, transition_fn, projection_fn, prediction_fn, params
        else:
            return preproc_fn, model_fn, transition_fn, projection_fn, prediction_fn

    return model_builder
