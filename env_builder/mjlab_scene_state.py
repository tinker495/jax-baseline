"""mjlab scene, actuator, terrain, and sensor state around evaluation."""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import TypeVar

import torch
import warp as wp
from mjlab.actuator.builtin_actuator import (
    BuiltinDcMotorActuator,
    BuiltinMotorActuator,
    BuiltinMuscleActuator,
    BuiltinPdActuator,
    BuiltinPositionActuator,
    BuiltinVelocityActuator,
)
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab.actuator.learned_actuator import LearnedMlpActuator
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sensor.builtin_sensor import BuiltinSensor
from mjlab.sensor.camera_sensor import CameraSensor
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.sensor.sensor import Sensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.terrains.terrain_entity import TerrainEntity
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.buffers.delay_buffer import DelayBuffer

SensorData = TypeVar("SensorData")


@contextmanager
def _preserve_tensors(tensors: Sequence[torch.Tensor | None]) -> Iterator[None]:
    saved = [(tensor, tensor.detach().clone()) for tensor in tensors if tensor is not None]
    try:
        yield
    finally:
        for tensor, value in saved:
            tensor.copy_(value)


@contextmanager
def _preserve_history(buffer: CircularBuffer) -> Iterator[None]:
    storage, pointer = buffer._buffer, buffer._pointer
    with _preserve_tensors((storage, buffer._num_pushes)):
        try:
            yield
        finally:
            buffer._buffer, buffer._pointer = storage, pointer


@contextmanager
def _preserve_learned_network(actuator: LearnedMlpActuator) -> Iterator[None]:
    network = actuator.network
    assert network is not None
    # TorchScript state includes ordinary attributes and parameters, beyond buffers.
    actuator.network = deepcopy(network)
    try:
        yield
    finally:
        actuator.network = network


@contextmanager
def _preserve_delay(delay: DelayBuffer) -> Iterator[None]:
    ring = delay._buffer
    buffer, pointer, lags = ring._buffer, ring._pointer, delay._current_lags
    generator_state = None if delay.generator is None else delay.generator.get_state()
    with _preserve_tensors(
        (buffer, ring._num_pushes, lags, delay._step_count, delay._phase_offsets)
    ):
        try:
            yield
        finally:
            ring._buffer, ring._pointer, delay._current_lags = buffer, pointer, lags
            if generator_state is not None:
                assert delay.generator is not None
                delay.generator.set_state(generator_state)


@contextmanager
def _preserve_contact(sensor: ContactSensor) -> Iterator[None]:
    cache, valid = sensor._cached_data, sensor._cache_valid
    tensors: list[torch.Tensor | None] = []
    if sensor._air_time_state is not None:
        tensors.extend(
            (
                sensor._air_time_state.current_air_time,
                sensor._air_time_state.last_air_time,
                sensor._air_time_state.current_contact_time,
                sensor._air_time_state.last_contact_time,
            )
        )
    history = None if sensor._history_state is None else sensor._history_state.copy()
    if history is not None:
        tensors.extend(history.values())
    with _preserve_tensors(tensors):
        try:
            yield
        finally:
            sensor._cached_data, sensor._cache_valid = cache, valid
            if history is not None:
                assert sensor._history_state is not None
                sensor._history_state.clear()
                sensor._history_state.update(history)


@contextmanager
def _preserve_height(sensor: RayCastSensor) -> Iterator[None]:
    cache, valid = sensor._cached_data, sensor._cache_valid
    references = (
        sensor._distances,
        sensor._normals_w,
        sensor._hit_pos_w,
        sensor._pos_w,
        sensor._quat_w,
        sensor._frame_pos_w,
        sensor._frame_quat_w,
        sensor._cached_world_origins,
        sensor._cached_world_rays,
        sensor._cached_frame_pos,
        sensor._cached_frame_mat,
    )
    ray_buffers = (
        sensor._ray_pnt,
        sensor._ray_vec,
        sensor._ray_dist,
        sensor._ray_geomid,
        sensor._ray_normal,
    )
    with _preserve_tensors(
        [
            *references,
            *(wp.to_torch(buffer) for buffer in ray_buffers if buffer is not None),
        ]
    ):
        try:
            yield
        finally:
            (
                sensor._distances,
                sensor._normals_w,
                sensor._hit_pos_w,
                sensor._pos_w,
                sensor._quat_w,
                sensor._frame_pos_w,
                sensor._frame_quat_w,
                sensor._cached_world_origins,
                sensor._cached_world_rays,
                sensor._cached_frame_pos,
                sensor._cached_frame_mat,
            ) = references
            sensor._cached_data, sensor._cache_valid = cache, valid


@contextmanager
def _preserve_builtin(sensor: Sensor[SensorData]) -> Iterator[None]:
    cache, valid = sensor._cached_data, sensor._cache_valid
    try:
        yield
    finally:
        sensor._cached_data, sensor._cache_valid = cache, valid


def validate_scene_state(scene: Scene) -> None:
    if type(scene) is not Scene:
        raise TypeError("State preservation does not support custom Scene classes")
    for entity in scene.entities.values():
        if type(entity) not in (Entity, TerrainEntity):
            raise TypeError(f"Unsupported entity type: {type(entity).__name__}")
        for actuator in entity.actuators:
            if type(actuator) not in (
                BuiltinPositionActuator,
                BuiltinPdActuator,
                BuiltinMotorActuator,
                BuiltinDcMotorActuator,
                BuiltinVelocityActuator,
                BuiltinMuscleActuator,
                XmlActuator,
                IdealPdActuator,
                DcMotorActuator,
                LearnedMlpActuator,
            ):
                raise TypeError(f"Unsupported actuator type: {type(actuator).__name__}")
    for sensor in scene.sensors.values():
        if isinstance(sensor, RayCastSensor) and any(
            buffer is None
            for buffer in (
                sensor._ray_pnt,
                sensor._ray_vec,
                sensor._ray_dist,
                sensor._ray_geomid,
                sensor._ray_normal,
            )
        ):
            raise ValueError("Ray sensors must be initialized before state preservation")
        if type(sensor) not in (
            BuiltinSensor,
            ContactSensor,
            RayCastSensor,
            TerrainHeightSensor,
            CameraSensor,
        ):
            raise TypeError(f"Unsupported sensor type: {type(sensor).__name__}")


@contextmanager
def preserve_scene_state(scene: Scene) -> Iterator[None]:
    """Preserve scene buffers; physics and global RNG are owned by the caller."""
    with ExitStack() as stack:
        stack.enter_context(_preserve_tensors((scene.env_origins,)))
        if scene.terrain is not None and scene.terrain.terrain_origins is not None:
            stack.enter_context(
                _preserve_tensors((scene.terrain.terrain_levels, scene.terrain.terrain_types))
            )
        if scene.sensor_context is not None:
            context = scene.sensor_context
            stack.enter_context(
                _preserve_tensors(
                    (
                        context._rgb_torch,
                        context._depth_torch,
                        context._seg_torch,
                        wp.to_torch(context.render_context.rgb_data),
                    )
                )
            )
        delays: dict[int, DelayBuffer] = {}
        for entity in scene.entities.values():
            data = entity.data
            stack.enter_context(
                _preserve_tensors(
                    (
                        data.default_root_state,
                        data.default_joint_pos,
                        data.default_joint_vel,
                        data.default_joint_pos_limits,
                        data.joint_pos_limits,
                        data.soft_joint_pos_limits,
                        data.gravity_vec_w,
                        data.forward_vec_b,
                        data.joint_pos_target,
                        data.joint_vel_target,
                        data.joint_effort_target,
                        data.tendon_len_target,
                        data.tendon_vel_target,
                        data.tendon_effort_target,
                        data.site_effort_target,
                        data.encoder_bias,
                    )
                )
            )
            for actuator in entity.actuators:
                if isinstance(actuator, IdealPdActuator):
                    stack.enter_context(
                        _preserve_tensors(
                            (
                                actuator.stiffness,
                                actuator.damping,
                                actuator.force_limit,
                                actuator.default_stiffness,
                                actuator.default_damping,
                                actuator.default_force_limit,
                            )
                        )
                    )
                if isinstance(actuator, DcMotorActuator):
                    stack.enter_context(
                        _preserve_tensors(
                            (
                                actuator.saturation_effort,
                                actuator.velocity_limit_motor,
                                actuator._joint_vel_clipped,
                            )
                        )
                    )
                if isinstance(actuator, LearnedMlpActuator):
                    stack.enter_context(_preserve_learned_network(actuator))
                    for history in (actuator._pos_error_history, actuator._vel_history):
                        if history is not None:
                            stack.enter_context(_preserve_history(history))
                if actuator._delay_buffer is not None:
                    delays[id(actuator._delay_buffer)] = actuator._delay_buffer
        for delay in delays.values():
            stack.enter_context(_preserve_delay(delay))
        for sensor in scene.sensors.values():
            if isinstance(sensor, RayCastSensor):
                stack.enter_context(_preserve_height(sensor))
            elif isinstance(sensor, ContactSensor):
                stack.enter_context(_preserve_contact(sensor))
            elif isinstance(sensor, (BuiltinSensor, CameraSensor)):
                stack.enter_context(_preserve_builtin(sensor))
        yield
