import os
import re
from collections import defaultdict, deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

gym.register_envs(ale_py)
os.environ.setdefault("PATH", "")
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial obses by taking random number of no-ops on reset. No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        # self._action_space = gym.spaces.Discrete(self.action_space.n-1)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env, kill_on_life_loss=False):
        """Make end-of-life == end-of-episode, but only reset on true game over. Done by DeepMind for the DQN and
        co. since it helps value estimation.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.kill_on_life_loss = kill_on_life_loss

    def true_reset(self, **kwargs):
        """Force a real environment reset, ignoring episodic life handling.

        This bypasses the loss-of-life pseudo-terminal behavior and resets the
        underlying environment, returning the initial observation and info.
        """
        obs, info = self.env.reset(**kwargs)
        # Mark last transition as a real episode end and refresh lives info
        self.was_real_done = True
        self.lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminated,
        # then update lives to handle bonus lives
        if self.was_real_done:
            info["lives"] = 0
        lives = info["lives"]
        if 0 < lives < self.lives:
            # print("Lives lost: ", self.lives - lives)
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises terminated.
            terminated = True
            # self.env.step(1)
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted. This way all obses are still reachable
        even though lives are episodic, and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: ([int] or [float]) the first observation of the environment
        """
        if self.was_real_done or self.kill_on_life_loss:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminated/lost life obs
            obs, _, _, _, info = self.env.step(0)
        self.lives = info["lives"]  # self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame (frameskipping)

        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action Repeat action, sum reward, and max over last observations.

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, terminated, information
        """
        total_reward = 0.0
        terminated = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the terminated=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """clips the reward to {+1, 0, -1} by its sign.

        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        """Logging the original reward."""

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
        return obs, self._reward(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_reward"] = 0
        return obs, info

    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign.

        :param reward: (float)
        """
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as terminated in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame.

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once. It exists purely
        to optimize memory usage which can be huge for DQN's 1M frames replay buffers. This object should only be
        converted to np.ndarray before being passed to the model.

        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(
    env,
    episode_life=True,
    kill_on_life_loss=False,
    clip_rewards=True,
    frame_stack=True,
    scale=False,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env, kill_on_life_loss)
    print("Action meaning : ", env.unwrapped.get_action_meanings())
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_wrap_atari(env_id="Breakout-v0", clip_rewards=False):
    # env = gym.make(env_id)
    env = make_atari(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)
    env = wrap_deepmind(env, clip_rewards=clip_rewards, frame_stack=True)
    return env


def get_env_type(env_id):
    _game_envs = defaultdict(set)

    # Re-parse the gym registry, since we could have new envs since last time.
    for name, env in gym.envs.registry.items():
        # print(env.entry_point, env.id)
        try:
            if "gymnasium" in env.entry_point:
                env_type = env.entry_point.split(".")[2].split(":")[0]
            elif "shimmy" in env.entry_point:
                env_type = env.entry_point.split(".")[1].split(":")[0]
            elif "ale_py" in env.entry_point:
                env_type = "atari_env"
            _game_envs[env_type].add(env.id)  # This is a set so add is idempotent
        except Exception:
            pass

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ":" in env_id:
            env_type = re.sub(r":.*", "", env_id)

    return env_type, env_id
