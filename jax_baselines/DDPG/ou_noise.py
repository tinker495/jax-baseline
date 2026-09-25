import jax


def ou_step(previous, key, theta=0.1, sigma=0.2):
    """One Ornstein-Uhlenbeck step, traced inside the DDPG behavior-action jits."""
    return previous - theta * previous + sigma * jax.random.normal(key, previous.shape)
