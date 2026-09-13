"""Enable final-video rendering inside the existing Gymnasium worker."""

from collections.abc import Iterator
from contextlib import contextmanager

import gymnasium as gym


class GymRenderingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        restore_state: bool = False,
        recording_worker: bool = True,
    ) -> None:
        super().__init__(env)
        self.supports_rendering = env.render_mode == "rgb_array" or (
            env.render_mode is None
            and type(env.unwrapped).__module__.startswith(
                (
                    "gymnasium.envs.classic_control.",
                    "gymnasium.envs.box2d.",
                    "gymnasium.envs.mujoco.",
                    "gymnasium.envs.toy_text.",
                )
            )
            and "rgb_array" in env.metadata["render_modes"]
        )
        self._restore_state = restore_state
        self._recording_worker = recording_worker
        self._rendering_active = False
        self._previous_render_mode = env.render_mode

    def start_rendering(self) -> None:
        if not self.supports_rendering:
            raise ValueError("This Gymnasium environment cannot enable RGB rendering in place")
        if self._rendering_active:
            raise RuntimeError("Environment is already borrowed for rendering")
        self._previous_render_mode = self.env.unwrapped.render_mode
        self._rendering_active = True
        if self._recording_worker and self._previous_render_mode != "rgb_array":
            self.env.unwrapped.render_mode = "rgb_array"

    def stop_rendering(self) -> None:
        if not self._rendering_active:
            return
        self._rendering_active = False
        if not self._recording_worker or self._previous_render_mode == "rgb_array":
            return
        try:
            if self._restore_state:
                # Release native resources before GymStateWrapper restores the
                # headless Python graph. Dedicated eval envs own theirs until close().
                self.env.unwrapped.close()
        finally:
            self.env.unwrapped.render_mode = self._previous_render_mode

    @contextmanager
    def rendering_context(self) -> Iterator[None]:
        self.start_rendering()
        try:
            yield
        finally:
            self.stop_rendering()

    def render(self):
        return self.env.render() if self._recording_worker else None
