import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.haiku.apply import get_apply_fn_haiku_module
from model_builder.haiku.layers import NoisyLinear
from model_builder.haiku.Module import PreProcess
from model_builder.utils import print_param


class Projection(hk.Module):
    def __init__(self, embed_size=128, node=256, hidden_n=2):
        super().__init__()
        self.embed_size = embed_size
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        feature = hk.Flatten()(feature)
        projection = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        projection = self.layer(self.embed_size)(projection)
        return projection


class Transition(hk.Module):
    def __init__(self, node=64, hidden_n=2):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = lambda ch: hk.Sequential(
            [
                hk.Conv2D(
                    ch,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                    w_init=hk.initializers.Orthogonal(scale=1.0),
                ),
                hk.GroupNorm(4),
                jax.nn.relu,
            ]
        )
        self.last = lambda ch: hk.Conv2D(
            ch,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding="SAME",
            w_init=hk.initializers.Orthogonal(scale=1.0),
        )

    def __call__(self, feature: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        batch_size = feature.shape[0]
        tile_shape = (1, feature.shape[1], feature.shape[2], 1)
        action = jnp.reshape(action, (batch_size, 1, 1, -1))
        action = jnp.tile(action, tile_shape)
        concat = jnp.concatenate([feature, action], axis=-1)
        x = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(concat)
        feature = self.layer(feature.shape[-1])(x)
        return feature


class Prediction(hk.Module):
    def __init__(self, node=256, hidden_n=1):
        super().__init__()
        self.node = node
        self.hidden_n = hidden_n
        self.layer = hk.Linear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        x = hk.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
        )(feature)
        feature = self.layer(feature.shape[-1])(x)
        return feature


class Model(hk.Module):
    def __init__(
        self,
        action_size,
        node=256,
        hidden_n=2,
        noisy=False,
        dueling=False,
        categorial_bar_n=51,
    ):
        super().__init__()
        self.action_size = action_size
        self.node = node
        self.hidden_n = hidden_n
        self.noisy = noisy
        self.dueling = dueling
        self.categorial_bar_n = categorial_bar_n
        if not noisy:
            self.layer = hk.Linear
        else:
            self.layer = NoisyLinear

    def __call__(self, feature: jnp.ndarray) -> jnp.ndarray:
        feature = hk.Flatten()(feature)
        if not self.dueling:
            q = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.categorial_bar_n,
                        w_init=hk.initializers.RandomUniform(-0.03, 0.03),
                    ),
                    hk.Reshape((self.action_size[0], self.categorial_bar_n)),
                ]
            )(feature)
            return jax.nn.softmax(q, axis=2)
        else:
            v = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.categorial_bar_n, w_init=hk.initializers.RandomUniform(-0.03, 0.03)
                    ),
                    hk.Reshape((1, self.categorial_bar_n)),
                ]
            )(feature)
            a = hk.Sequential(
                [
                    self.layer(self.node) if i % 2 == 0 else jax.nn.relu
                    for i in range(2 * self.hidden_n)
                ]
                + [
                    self.layer(
                        self.action_size[0] * self.categorial_bar_n,
                        w_init=hk.initializers.RandomUniform(-0.03, 0.03),
                    ),
                    hk.Reshape((self.action_size[0], self.categorial_bar_n)),
                ]
            )(feature)
            q = v + a - jnp.mean(a, axis=1, keepdims=True)
            return jax.nn.softmax(q, axis=2)


def model_builder_maker(
    observation_space, action_space, dueling_model, param_noise, categorial_bar_n, policy_kwargs
):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def _model_builder(key=None, print_model=False):
        preproc = hk.transform(
            lambda x: hk.Sequential(
                [
                    PreProcess(observation_space, embedding_mode=embedding_mode),
                    hk.Reshape((7, 7, 64)),
                ]
            )(x)
        )
        model = hk.transform(
            lambda x: Model(
                action_space,
                dueling=dueling_model,
                noisy=param_noise,
                categorial_bar_n=categorial_bar_n,
                **policy_kwargs
            )(x)
        )
        transition = hk.transform(lambda x, y: Transition()(x, y))
        projection = hk.transform(lambda x: Projection()(x))
        prediction = hk.transform(lambda x: Prediction()(x))
        preproc_fn = get_apply_fn_haiku_module(preproc)
        model_fn = get_apply_fn_haiku_module(model)
        transition_fn = get_apply_fn_haiku_module(transition)
        projection_fn = get_apply_fn_haiku_module(projection)
        prediction_fn = get_apply_fn_haiku_module(prediction)
        if key is not None:
            keys = jax.random.split(key, 8)
            pre_param = preproc.init(
                keys[0],
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            feature = preproc.apply(
                pre_param, keys[1], [np.zeros((1, *o), dtype=np.float32) for o in observation_space]
            )
            model_param = model.init(keys[2], feature)
            action = jnp.zeros((1, action_space[0]), dtype=np.float32)
            transition_param = transition.init(keys[3], feature, action)
            transition_feature = transition.apply(transition_param, keys[4], feature, action)
            projection_param = projection.init(keys[5], transition_feature)
            projection_feature = projection.apply(projection_param, keys[6], transition_feature)
            prediction_param = prediction.init(keys[7], projection_feature)
            params = hk.data_structures.merge(
                pre_param, model_param, transition_param, projection_param, prediction_param
            )
            if print_model:
                print("------------------build-haiku-model--------------------")
                print_param("preprocess", pre_param)
                print_param("model", model_param)
                print_param("transition", transition_param)
                print_param("projection", projection_param)
                print_param("prediction", prediction_param)
                print("-------------------------------------------------------")
            return preproc_fn, model_fn, transition_fn, projection_fn, prediction_fn, params
        else:
            return preproc_fn, model_fn, transition_fn, projection_fn, prediction_fn

    return _model_builder
