"""Explicit state adapters for mjlab 1.6 action, command, and manager terms."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import FunctionType

from mjlab.envs.mdp.actions.actions import (
    BaseAction,
    JointEffortAction,
    JointPositionAction,
    JointVelocityAction,
    RelativeJointPositionAction,
    SiteEffortAction,
    TendonEffortAction,
    TendonLengthAction,
    TendonVelocityAction,
)
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKAction
from mjlab.envs.mdp.curriculums import reward_curriculum, termination_curriculum
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.envs.mdp.rewards import electrical_power_cost, posture
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.manipulation.mdp.commands import (
    LiftingCommand,
    MultiCubeLiftingCommand,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.velocity.mdp.rewards import (
    feet_swing_height,
    upright,
    variable_posture,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.noise.noise_cfg import (
    ConstantNoiseCfg,
    GaussianNoiseCfg,
    NoiseCfg,
    UniformNoiseCfg,
)
from mjlab.utils.noise.noise_model import NoiseModel, NoiseModelWithAdditiveBias

from env_builder.mjlab_scene_state import _preserve_tensors
from jax_baselines.core.env_protocols import EvaluationContextEnv

_ACTION_TYPES = (
    JointPositionAction,
    RelativeJointPositionAction,
    JointVelocityAction,
    JointEffortAction,
    TendonLengthAction,
    TendonVelocityAction,
    TendonEffortAction,
    SiteEffortAction,
    DifferentialIKAction,
)
_COMMAND_TYPES = (
    UniformVelocityCommand,
    MotionCommand,
    LiftingCommand,
    MultiCubeLiftingCommand,
)
_CALLABLE_TYPES = (
    feet_swing_height,
    upright,
    variable_posture,
    posture,
    electrical_power_cost,
    apply_body_impulse,
    reward_curriculum,
    termination_curriculum,
)
_NOISE_TYPES = (ConstantNoiseCfg, UniformNoiseCfg, GaussianNoiseCfg)
# These installed functions mutate only the env/scene/model or manager buffers
# covered by the snapshot. User callables must declare their own state context.
_FUNCTION_MODULES = {
    "mjlab.envs.mdp.observations",
    "mjlab.envs.mdp.rewards",
    "mjlab.envs.mdp.terminations",
    "mjlab.envs.mdp.events",
    "mjlab.envs.mdp.metrics",
    "mjlab.tasks.cartpole.cartpole_env_cfg",
    *(
        f"mjlab.envs.mdp.dr.{name}"
        for name in (
            "actuator",
            "body",
            "camera",
            "geom",
            "joint",
            "light",
            "material",
            "pair",
            "site",
            "tendon",
        )
    ),
    *(
        f"mjlab.tasks.{task}.mdp.{module}"
        for task, modules in (
            ("velocity", ("observations", "rewards", "terminations", "curriculums")),
            ("manipulation", ("observations", "rewards", "terminations")),
            ("tracking", ("observations", "rewards", "terminations", "metrics")),
        )
        for module in modules
    ),
}


def validate_term_state(term: object) -> None:
    """Accept audited built-ins or a custom term with an explicit state context."""
    if isinstance(term, EvaluationContextEnv):
        return
    if isinstance(term, FunctionType) and term.__module__ in _FUNCTION_MODULES:
        return
    if type(term) not in (
        *_ACTION_TYPES,
        *_COMMAND_TYPES,
        *_CALLABLE_TYPES,
        *_NOISE_TYPES,
        NoiseModel,
        NoiseModelWithAdditiveBias,
    ):
        raise TypeError(
            f"Unsupported mjlab term {type(term).__name__}; custom terms must implement "
            "evaluation_context() to preserve their mutable state"
        )
    if isinstance(term, UniformVelocityCommand) and term._joystick_enabled is not None:
        raise ValueError("State preservation does not support interactive command overrides")
    if isinstance(term, NoiseModel):
        validate_term_state(term._noise_model_cfg.noise_cfg)
    if isinstance(term, NoiseModelWithAdditiveBias):
        validate_term_state(term._bias_noise_cfg)
    if isinstance(term, reward_curriculum) and any(
        set(stage) - {"step", "weight", "params"} for stage in term._stages
    ):
        raise ValueError("Reward curricula may change only weight and params")
    if isinstance(term, termination_curriculum) and any(
        set(stage) - {"step", "time_out", "params"} for stage in term._stages
    ):
        raise ValueError("Termination curricula may change only time_out and params")


@contextmanager
def _preserve_command(term: CommandTerm) -> Iterator[None]:
    metrics = term.metrics.copy()
    with _preserve_tensors((term.time_left, term.command_counter, *metrics.values())):
        try:
            if isinstance(term, UniformVelocityCommand):
                heading = term.heading_error
                ranges = (
                    term.cfg.ranges.lin_vel_x,
                    term.cfg.ranges.lin_vel_y,
                    term.cfg.ranges.ang_vel_z,
                    term.cfg.ranges.heading,
                )
                with _preserve_tensors(
                    (
                        term.vel_command_b,
                        term.vel_command_w,
                        term.heading_target,
                        heading,
                        term.is_heading_env,
                        term.is_standing_env,
                        term.is_world_env,
                        term.is_forward_env,
                    )
                ):
                    try:
                        yield
                    finally:
                        term.heading_error = heading
                        (
                            term.cfg.ranges.lin_vel_x,
                            term.cfg.ranges.lin_vel_y,
                            term.cfg.ranges.ang_vel_z,
                            term.cfg.ranges.heading,
                        ) = ranges
                return
            if isinstance(term, MotionCommand):
                poses = term.body_pos_relative_w, term.body_quat_relative_w
                failed, pending = term.bin_failed_count, term._pending_forward
                with _preserve_tensors((term.time_steps, *poses, failed, term._current_bin_failed)):
                    try:
                        yield
                    finally:
                        term.body_pos_relative_w, term.body_quat_relative_w = poses
                        term.bin_failed_count, term._pending_forward = failed, pending
                return
            if isinstance(term, (LiftingCommand, MultiCubeLiftingCommand)):
                success, pending = term.episode_success, term._pending_forward
                with ExitStack() as stack:
                    stack.enter_context(_preserve_tensors((term.target_pos, success)))
                    if isinstance(term, MultiCubeLiftingCommand):
                        target = term._cached_target_obj_pos
                        stack.enter_context(_preserve_tensors((term.target_selection, target)))
                        try:
                            yield
                        finally:
                            term._cached_target_obj_pos = target
                            term.episode_success, term._pending_forward = (
                                success,
                                pending,
                            )
                    else:
                        try:
                            yield
                        finally:
                            term.episode_success, term._pending_forward = (
                                success,
                                pending,
                            )
                return
            raise TypeError(f"Unsupported command: {type(term).__name__}")
        finally:
            term.metrics.clear()
            term.metrics.update(metrics)


@contextmanager
def preserve_term_state(term: object) -> Iterator[None]:
    """Preserve a term accepted by validate_term_state, including rebound tensors."""
    if isinstance(term, EvaluationContextEnv):
        with term.evaluation_context():
            yield
        return
    if isinstance(term, CommandTerm):
        with _preserve_command(term):
            yield
        return
    if isinstance(term, BaseAction):
        processed = term._processed_actions
        with _preserve_tensors((term._raw_actions, processed)):
            try:
                yield
            finally:
                term._processed_actions = processed
        return
    if isinstance(term, DifferentialIKAction):
        with _preserve_tensors(
            (
                term._raw_actions,
                term._desired_pos,
                term._desired_quat,
                term._jacp_torch,
                term._jacr_torch,
                term._point_torch,
            )
        ):
            yield
        return
    if isinstance(term, feet_swing_height):
        peak = term.peak_heights
        with _preserve_tensors((peak,)):
            try:
                yield
            finally:
                term.peak_heights = peak
        return
    if isinstance(term, apply_body_impulse):
        with _preserve_tensors((term._time_remaining, term._active, term._interval_time_left)):
            yield
        return
    if isinstance(term, NoiseCfg):
        cache = {device: values.copy() for device, values in term._tensor_cache.items()}
        try:
            yield
        finally:
            term._tensor_cache.clear()
            term._tensor_cache.update(cache)
        return
    if isinstance(term, NoiseModel):
        with ExitStack() as stack:
            stack.enter_context(preserve_term_state(term._noise_model_cfg.noise_cfg))
            if isinstance(term, NoiseModelWithAdditiveBias):
                bias, components, initialized = (
                    term._bias,
                    term._num_components,
                    term._bias_initialized,
                )
                stack.enter_context(preserve_term_state(term._bias_noise_cfg))
                stack.enter_context(_preserve_tensors((bias,)))
                try:
                    yield
                finally:
                    term._bias, term._num_components, term._bias_initialized = (
                        bias,
                        components,
                        initialized,
                    )
            else:
                yield
        return
    # Built-in posture/reward callables have construction-only tensor constants;
    # curriculum target config mutations are restored by the manager snapshot.
    yield
