"""Validated, immutable network descriptions shared by both model backends."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "sigmoid": jax.nn.sigmoid,
    "leaky_relu": jax.nn.leaky_relu,
    "identity": lambda x: x,
}
EMBEDDING_MODES = ("normal", "resnet")


@dataclass(frozen=True)
class LayerConfig:
    units: int
    activation: str = "relu"

    def __post_init__(self):
        if type(self.units) is not int or self.units < 1:
            raise ValueError("Layer units must be a positive integer")
        if not isinstance(self.activation, str) or self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {self.activation!r}; use {sorted(ACTIVATIONS)}")


@dataclass(frozen=True)
class MLPConfig:
    layers: tuple[LayerConfig, ...] = (LayerConfig(256),) * 2
    embedding_mode: str = "normal"

    def __post_init__(self):
        if not isinstance(self.layers, tuple) or any(
            not isinstance(layer, LayerConfig) for layer in self.layers
        ):
            raise ValueError("MLP layers must be a tuple of LayerConfig values")
        if not isinstance(self.embedding_mode, str) or self.embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"Unknown embedding mode: {self.embedding_mode!r}")


@dataclass(frozen=True)
class ResidualConfig:
    kind: Literal["simba", "simbav2", "flashsac"]
    blocks: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    embedding_mode: str = "normal"

    def __post_init__(self):
        if self.kind not in ("simba", "simbav2", "flashsac"):
            raise ValueError(f"Unknown residual network type: {self.kind!r}")
        if (
            not isinstance(self.blocks, tuple)
            or not self.blocks
            or any(type(width) is not int or width < 1 for width in self.blocks)
        ):
            raise ValueError("Residual blocks must contain at least one positive integer width")
        if not isinstance(self.activation, str) or self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {self.activation!r}; use {sorted(ACTIVATIONS)}")
        if not isinstance(self.embedding_mode, str) or self.embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"Unknown embedding mode: {self.embedding_mode!r}")


ModelConfig = MLPConfig | ResidualConfig
DEFAULT_MLP = MLPConfig()


def parse_model_config(value: object) -> ModelConfig:
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError("Model JSON must be an object with a 'type' field")
    options: dict[str, object] = {"activation": "relu", "embedding_mode": "normal"}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Model JSON keys must be strings")
        options[key] = item
    embedding_mode = options["embedding_mode"]
    if not isinstance(embedding_mode, str):
        raise TypeError("Model embedding_mode must be a string")
    if options["type"] == "mlp":
        if (
            set(options) - {"type", "layers", "activation", "embedding_mode"}
            or "layers" not in options
        ):
            raise ValueError(
                "MLP JSON requires 'layers' and accepts type/layers/activation/embedding_mode"
            )
        sizes = options["layers"]
        if not isinstance(sizes, list):
            raise ValueError("MLP layers must be a JSON array of positive integers")
        activation = options["activation"]
        if isinstance(activation, str):
            if activation not in ACTIVATIONS:
                raise ValueError(f"Unknown activation: {activation!r}")
            activation = [activation] * len(sizes)
        if not isinstance(activation, list) or len(activation) != len(sizes):
            raise ValueError("MLP activation must be a name or one name per hidden layer")
        layers = []
        for size, name in zip(sizes, activation, strict=True):
            if type(size) is not int or not isinstance(name, str):
                raise ValueError("Each MLP layer requires integer units and an activation name")
            layers.append(LayerConfig(size, name))
        return MLPConfig(tuple(layers), embedding_mode=embedding_mode)
    kinds: dict[str, Literal["simba", "simbav2", "flashsac"]] = {
        "simba": "simba",
        "simbav2": "simbav2",
        "flashsac": "flashsac",
    }
    kind = options["type"]
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError(f"Unknown model type: {kind!r}")
    if set(options) - {"type", "blocks", "activation", "embedding_mode"} or "blocks" not in options:
        raise ValueError(
            "Residual JSON requires 'blocks' and accepts type/blocks/activation/embedding_mode"
        )
    blocks, activation = options["blocks"], options["activation"]
    if not isinstance(blocks, list) or not isinstance(activation, str):
        raise TypeError("Residual blocks must be a JSON array and activation must be a name")
    widths = []
    for width in blocks:
        if type(width) is not int:
            raise TypeError("Residual block widths must be integers")
        widths.append(width)
    return ResidualConfig(kinds[kind], tuple(widths), activation, embedding_mode=embedding_mode)


def load_model_config(path: str | Path) -> ModelConfig:
    try:
        return parse_model_config(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model config {path}: {exc}") from exc


def model_config_dict(config: ModelConfig) -> dict:
    if isinstance(config, MLPConfig):
        names = [layer.activation for layer in config.layers]
        return {
            "type": "mlp",
            "layers": [layer.units for layer in config.layers],
            "activation": names[0] if names and len(set(names)) == 1 else names,
            "embedding_mode": config.embedding_mode,
        }
    return {
        "type": config.kind,
        "blocks": list(config.blocks),
        "activation": config.activation,
        "embedding_mode": config.embedding_mode,
    }


def resolve_model_config(
    value: object,
    default: ModelConfig,
    *,
    allowed_types: tuple[str, ...] | None = None,
    allowed_embeddings: tuple[str, ...] = EMBEDDING_MODES,
) -> ModelConfig:
    if value is None:
        config = default
    elif isinstance(value, (str, Path)):
        config = load_model_config(value)
    elif isinstance(value, (MLPConfig, ResidualConfig)):
        config = value
    else:
        config = parse_model_config(value)
    if allowed_types is None:
        allowed_types = ("mlp" if isinstance(default, MLPConfig) else default.kind,)
    actual = "mlp" if isinstance(config, MLPConfig) else config.kind
    if actual not in allowed_types:
        raise ValueError(f"This builder supports model types {allowed_types}, received {actual!r}")
    if config.embedding_mode not in allowed_embeddings:
        raise ValueError(
            f"This builder supports embedding modes {allowed_embeddings}, "
            f"received {config.embedding_mode!r}"
        )
    return config
