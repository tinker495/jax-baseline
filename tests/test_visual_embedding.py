import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from model_builder.flax.Module import visual_embedding as flax_visual_embedding
from model_builder.haiku.Module import visual_embedding as haiku_visual_embedding

_IMAGE = jnp.zeros((1, 84, 84, 4), dtype=jnp.float32)


def _flax_output(mode, *, multiple=1):
    model = flax_visual_embedding(mode, multiple=multiple)
    params = model.init(jax.random.PRNGKey(0), _IMAGE)
    return model.apply(params, _IMAGE)


def _haiku_output(mode=None):
    def forward(image):
        embed = haiku_visual_embedding() if mode is None else haiku_visual_embedding(mode)
        return embed(image)

    model = hk.without_apply_rng(hk.transform(forward))
    params = model.init(jax.random.PRNGKey(0), _IMAGE)
    return model.apply(params, _IMAGE)


def test_flax_normal_visual_embedding_shape():
    assert _flax_output("normal").shape == (1, 3136)


def test_flax_resnet_visual_embedding_shape():
    assert _flax_output("resnet", multiple=4).shape == (1, 15488)


def test_haiku_default_visual_embedding_is_normal():
    assert _haiku_output().shape == (1, 3136)


@pytest.mark.parametrize("mode", ["simple", "minimum", "none"])
def test_unused_flax_visual_embedding_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Unknown visual_embedding mode"):
        flax_visual_embedding(mode)


@pytest.mark.parametrize("mode", ["simple", "minimum", "none"])
def test_unused_haiku_visual_embedding_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Unknown visual_embedding mode"):
        haiku_visual_embedding(mode)
