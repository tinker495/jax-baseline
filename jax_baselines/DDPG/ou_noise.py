import jax
import jax.numpy as jnp
import numpy as np


@jax.jit(static_argnames=("theta", "sigma"))
def _device_noise(previous, key, theta, sigma):
    key, sample_key = jax.random.split(key)
    return previous - theta * previous + sigma * jax.random.normal(sample_key, previous.shape), key


class OUNoise:
    def __init__(self, sigma=0.2, theta=0.1, action_size=1, worker_size=1, key=None):
        self._theta = theta
        self._sigma = sigma
        self.action_size = action_size
        self.worker_size = worker_size
        self.key = key
        self.noise_prev = (
            sigma * jax.random.normal(key, (worker_size, action_size))
            if key is not None
            else np.random.normal(0, self._sigma, size=(self.worker_size, self.action_size))
        )

    def __call__(self):
        if self.key is not None:
            self.noise_prev, self.key = _device_noise(
                self.noise_prev, self.key, self._theta, self._sigma
            )
            return self.noise_prev
        noise = (
            self.noise_prev
            - self._theta * self.noise_prev
            + np.random.normal(0, self._sigma, size=(self.worker_size, self.action_size))
        )
        self.noise_prev = noise
        return noise

    def reset(self, worker) -> None:
        if self.key is not None:
            self.key, sample_key = jax.random.split(self.key)
            self.noise_prev = (
                jnp.asarray(self.noise_prev)
                .at[worker]
                .set(self._sigma * jax.random.normal(sample_key, (len(worker), self.action_size)))
            )
            return
        self.noise_prev[worker] = np.random.normal(
            0, self._sigma, size=(len(worker), self.action_size)
        )
