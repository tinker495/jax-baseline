"""Borrow Gymnasium environments by restoring their existing state objects.

Python instance dictionaries form the state protocol; slots and opaque native
objects require an environment-owned ``evaluation_context``. Code, classes and
module globals are configuration, while instance-owned containers and arrays are
restored in place. MuJoCo model/data buffers use the engine's native copy API.
"""

import ctypes
import inspect
import random
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from functools import lru_cache
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from types import BuiltinFunctionType, CellType, FunctionType, MethodType, ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard

import gymnasium as gym
import numpy as np

from jax_baselines.core.env_protocols import EvaluationContextEnv

if TYPE_CHECKING:
    from gymnasium.envs.mujoco import MujocoEnv


def _is_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return type(value) is dict


def _is_set(value: object) -> TypeGuard[set[object]]:
    return type(value) is set


class _PythonState:
    """Memoized undo log over validated native Python serialization dictionaries."""

    def __init__(self, external: tuple[object, ...] = ()):
        self.stack = ExitStack()
        self.seen = {id(value) for value in external}

    def capture(self, value: object, path: str) -> None:
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if isinstance(value, np.void):
            raise TypeError(f"{path}: mutable NumPy record scalars need an evaluation_context")
        if value is None or isinstance(
            value,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                range,
                slice,
                np.generic,
                np.dtype,
                type,
                ModuleType,
            ),
        ):
            return
        if isinstance(value, FunctionType):
            self.capture(value.__defaults__, path + ".defaults")
            self.capture(value.__kwdefaults__, path + ".keyword_defaults")
            self.capture(value.__closure__, path + ".closure")
            self.capture(object.__getstate__(value), path + ".state")
            return
        if isinstance(value, BuiltinFunctionType):
            self.capture(value.__self__, path + ".owner")
            return
        if isinstance(value, MethodType):
            self.capture(value.__self__, path + ".owner")
            return
        if isinstance(value, CellType):
            try:
                content = value.cell_contents
            except ValueError:

                def restore_empty_cell() -> None:
                    del value.cell_contents

                self.stack.callback(restore_empty_cell)
                return

            def restore_cell() -> None:
                value.cell_contents = content

            self.stack.callback(restore_cell)
            self.capture(content, path + ".content")
            return
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray:
                raise TypeError(f"{path}: NumPy subclasses need an evaluation_context")
            saved = value.copy()
            writable = value.flags.writeable
            self.capture(value.base, path + ".base")
            if value.dtype.hasobject:
                for index, item in enumerate(value.flat):
                    self.capture(item, f"{path}[{index}]")

            def restore_array() -> None:
                if value.shape != saved.shape or value.dtype != saved.dtype:
                    raise RuntimeError(f"{path}: evaluation changed an array's storage layout")
                if value.tobytes() != saved.tobytes():
                    if not value.flags.writeable:
                        value.setflags(write=True)
                    np.copyto(value, saved)
                if value.flags.writeable != writable:
                    value.setflags(write=writable)

            self.stack.callback(restore_array)
            return
        if type(value) is np.random.Generator:
            generator_state = deepcopy(value.bit_generator.state)

            def restore_generator() -> None:
                value.bit_generator.state = generator_state

            self.stack.callback(restore_generator)
            return
        if type(value) is np.random.RandomState:
            self.stack.callback(value.set_state, value.get_state())
            return
        if type(value) is random.Random:
            self.stack.callback(value.setstate, value.getstate())
            return
        if _is_mapping(value):
            saved_mapping = value.copy()

            def restore_mapping() -> None:
                value.clear()
                value.update(saved_mapping)

            self.stack.callback(restore_mapping)
            for key, item in saved_mapping.items():
                self.capture(key, path + ".key")
                self.capture(item, f"{path}[{key!r}]")
            return
        if type(value) is list or type(value) is deque:
            saved_items = list(value)
            self.stack.callback(value.extend, saved_items)
            self.stack.callback(value.clear)
            for index, item in enumerate(saved_items):
                self.capture(item, f"{path}[{index}]")
            return
        if _is_set(value):
            saved_members = value.copy()

            def restore_set() -> None:
                value.clear()
                value.update(saved_members)

            self.stack.callback(restore_set)
            for item in value:
                self.capture(item, path + ".member")
            return
        if type(value) is tuple or type(value) is frozenset:
            for index, item in enumerate(value):
                self.capture(item, f"{path}[{index}]")
            return
        # Reject extension bases even when a Python subclass exposes a dictionary:
        # its native buffers are absent from the default serialization state.
        for cls in type(value).__mro__:
            if cls is object:
                continue
            try:
                source = inspect.getsourcefile(cls)
            except TypeError as error:
                raise TypeError(
                    f"{path}: native {type(value).__name__} needs an evaluation_context"
                ) from error
            if source is None:
                raise TypeError(
                    f"{path}: native {type(value).__name__} needs an evaluation_context"
                )
        # Bypass EzPickle.__getstate__, which only stores constructor arguments.
        # This is Python's native serialization mapping, not dynamic field access.
        state = object.__getstate__(value)
        if state is not None and (
            type(state) is not dict or any(not isinstance(key, str) for key in state)
        ):
            raise TypeError(f"{path}: non-dictionary instance state needs an evaluation_context")
        if state is None:

            def restore_empty() -> None:
                current = object.__getstate__(value)
                if current is not None:
                    if type(current) is not dict:
                        raise TypeError(f"{path}: evaluation introduced non-dictionary state")
                    current.clear()

            self.stack.callback(restore_empty)
            return
        self.capture(state, path + ".state")


@contextmanager
def preserve_gym_spaces(*spaces: gym.Space) -> Iterator[None]:
    """Preserve vector-parent spaces, including all nested space generators."""
    snapshot = _PythonState()
    for index, space in enumerate(spaces):
        snapshot.capture(space, f"space[{index}]")
    try:
        yield
    finally:
        snapshot.stack.close()


@lru_cache(maxsize=1)
def _copy_model_function() -> Callable[[int, int], int]:
    libraries = [
        path for path in files("mujoco").iterdir() if path.name.startswith("libmujoco.so.")
    ]
    if len(libraries) != 1:
        raise RuntimeError("Expected one installed MuJoCo shared library for native state copying")
    function = ctypes.CDLL(str(libraries[0])).mj_copyModel
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_void_p
    return function


@contextmanager
def _preserve_mujoco(env: "MujocoEnv") -> Iterator[None]:
    if TYPE_CHECKING:
        # Upstream C bindings omit this export's Python declaration.
        def mj_copyData(destination: Any, model: Any, source: Any) -> None:
            ...

    else:
        from mujoco import mj_copyData

    model, data = env.model, env.data
    saved_model, saved_data = copy(model), copy(data)
    try:
        yield
    finally:
        if _copy_model_function()(model._address, saved_model._address) != model._address:
            raise RuntimeError("MuJoCo failed to restore the existing model buffer")
        mj_copyData(data, model, saved_data)


class GymStateWrapper(gym.Wrapper):
    """Keep worker-local snapshots without constructing another environment.

    Custom native environments can implement ``EvaluationContextEnv`` and own
    their simulator restoration. Their surrounding Python wrappers and spaces
    still participate in this snapshot. Environment topology must stay fixed.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if version("gymnasium") != "1.3.0":
            raise ValueError("Training-env evaluation requires Gymnasium 1.3.0")
        if env.render_mode is not None:
            raise ValueError("Training-env evaluation requires a headless Gymnasium environment")
        self._native_context = (
            env.unwrapped if isinstance(env.unwrapped, EvaluationContextEnv) else None
        )
        self._mujoco: MujocoEnv | None = None
        if find_spec("mujoco") is not None:
            from gymnasium.envs.mujoco import MujocoEnv as NativeMujocoEnv

            if isinstance(env.unwrapped, NativeMujocoEnv):
                self._mujoco = env.unwrapped
        if self._mujoco is not None and self._native_context is None:
            if version("mujoco") != "3.8.1":
                raise ValueError("Training-env evaluation requires MuJoCo 3.8.1")
            if self._mujoco.model.nplugin:
                raise ValueError("MuJoCo plugins require an environment-owned evaluation_context")
            _copy_model_function()
        self._evaluation_stack: ExitStack | None = None
        # Also validate the currently initialized graph at construction. Lazy task
        # state is checked while taking the snapshot, before evaluation can reset.
        snapshot = self._python_snapshot()
        snapshot.stack.close()

    def _python_snapshot(self) -> _PythonState:
        external: tuple[object, ...] = ()
        if self._native_context is not None:
            external = (self._native_context,)
        elif self._mujoco is not None:
            external = (
                self._mujoco.model,
                self._mujoco.data,
                self._mujoco.model.body_mass.base,
                self._mujoco.data.qpos.base,
            )
        snapshot = _PythonState(external)
        snapshot.capture(self.env, "env")
        if self._native_context is not None:
            snapshot.capture(self.action_space, "action_space")
            snapshot.capture(self.observation_space, "observation_space")
        return snapshot

    def save_training_state(self) -> None:
        if self._evaluation_stack is not None:
            raise RuntimeError("Training environment is already borrowed for evaluation")
        snapshot = self._python_snapshot()
        with ExitStack() as stack:
            stack.callback(random.setstate, random.getstate())
            stack.callback(np.random.set_state, np.random.get_state())
            if self._native_context is not None:
                stack.enter_context(self._native_context.evaluation_context())
            elif self._mujoco is not None:
                stack.enter_context(_preserve_mujoco(self._mujoco))
            stack.callback(snapshot.stack.close)
            self._evaluation_stack = stack.pop_all()

    def restore_training_state(self) -> None:
        if self._evaluation_stack is None:
            raise RuntimeError("No saved training environment state")
        stack, self._evaluation_stack = self._evaluation_stack, None
        stack.close()

    @contextmanager
    def evaluation_context(self) -> Iterator[None]:
        self.save_training_state()
        try:
            yield
        finally:
            self.restore_training_state()
