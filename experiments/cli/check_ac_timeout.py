"""Check timeout bootstrap invariance through a real Gym adapter and AC rollout."""

import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from env_builder.env_builder import get_env_builder
from experiments.optimizers import make_optimizer_factory
from jax_baselines.A2C.a2c import A2C
from jax_baselines.core.runtime_adapters import NoOpLoggerRun
from jax_baselines.core.training_session import RunContext
from jax_baselines.math.returns import discount_with_terminated, get_gaes


class TimeoutEnv(gym.Env):
    """One-step timeout with shared observation storage overwritten by reset."""

    def __init__(self, reset_value, render_mode=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation = np.zeros(1, np.float32)
        self.reset_value = reset_value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation.fill(self.reset_value)
        return self.observation, {}

    def step(self, action):
        self.observation.fill(100)
        return self.observation, 1.0, False, True, {}


class TimeoutAgent(A2C):
    """Deterministic actions isolate preprocessing from policy initialization."""

    def action_discrete(self, obs, eval=False):
        return np.zeros((1, 1), np.int32)


def main():
    for reset_value in (2, 1000):
        env_id = f"AC-Timeout-{reset_value}-v0"
        gym.register(
            env_id,
            entry_point="experiments.cli.check_ac_timeout:TimeoutEnv",
            kwargs={"reset_value": reset_value},
        )
        agent = TimeoutAgent(
            get_env_builder(env_id)[0],
            None,
            _init_setup_model=False,
            obs_normalization=True,
            optimizer_factory=make_optimizer_factory("adam"),
        )
        try:
            agent.learn_SingleEnv(RunContext(NoOpLoggerRun("."), 100, range(1, 2), 100))
            batch = agent.buffer.get_buffer()
            # An identity critic makes the effect on both bootstrap targets explicit.
            successor = jnp.asarray(batch["nxtobses"]["unified_obs"][0])
            rewards = jnp.asarray(batch["rewards"][0])
            terminateds = jnp.asarray(batch["terminateds"][0])
            truncateds = jnp.asarray(batch["truncateds"][0])
            a2c_target = discount_with_terminated(
                rewards, terminateds, truncateds, successor, agent.gamma
            )
            gae = get_gaes(
                rewards,
                terminateds,
                truncateds,
                jnp.zeros_like(successor),
                successor,
                agent.gamma,
                0.95,
            )
            print(
                f"reset={reset_value} successor={float(successor[0, 0]):.6f} "
                f"a2c_target={float(a2c_target[0, 0]):.6f} gae={float(gae[0, 0]):.6f}"
            )
            np.testing.assert_allclose(successor, 100 / 1.01, rtol=1e-6)
            np.testing.assert_allclose(a2c_target, 1 + agent.gamma * 100 / 1.01, rtol=1e-6)
            np.testing.assert_allclose(gae, a2c_target, rtol=1e-6)
        finally:
            agent.env.close()
            agent.eval_env.close()
    print("PASS: timeout successors and bootstrap targets are independent of reset samples.")


if __name__ == "__main__":
    main()
