import jax.numpy as jnp


def convert_normalized_obs(obs: dict):
    """Cast observations to float32, scaling compact uint8 pixels to [0, 1]."""
    return {
        key: value.astype(jnp.float32)
        if value.dtype != jnp.uint8
        else value.astype(jnp.float32) / 255.0
        for key, value in obs.items()
    }


if __name__ == "__main__":
    normalized = convert_normalized_obs(
        {"pixels": jnp.array([0, 255], dtype=jnp.uint8), "state": jnp.array([-2.0, 2.0])}
    )
    assert all(value.dtype == jnp.float32 for value in normalized.values())
    assert normalized["pixels"].tolist() == [0.0, 1.0]
    assert normalized["state"].tolist() == [-2.0, 2.0]
    print("PASS: uint8 [0, 255] -> float32 [0, 1]; float values unchanged.")
