"""Version-scoped mjlab state preservation for evaluation on the training env.

GPU simulation is not bitwise deterministic. Restoring transition inputs does not
guarantee an identical future trajectory. The caller preserves adapter observations
and pending results; custom terms own their state through evaluation_context().
"""

import random
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.metadata import version

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.command_manager import CommandManager, NullCommandManager
from mjlab.managers.curriculum_manager import CurriculumManager, NullCurriculumManager
from mjlab.managers.event_manager import EventManager
from mjlab.managers.manager_base import ManagerTermBaseCfg
from mjlab.managers.metrics_manager import MetricsManager, NullMetricsManager
from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.recorder_manager import NullRecorderManager
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.buffers.delay_buffer import DelayBuffer
from mjlab.utils.noise.noise_cfg import NoiseCfg

from env_builder.mjlab_physics_state import (
    preserve_physics_state,
    validate_physics_state,
)
from env_builder.mjlab_scene_state import (
    _preserve_delay,
    _preserve_history,
    _preserve_tensors,
    preserve_scene_state,
    validate_scene_state,
)
from env_builder.mjlab_term_state import preserve_term_state, validate_term_state


def _manager_terms(env: ManagerBasedRlEnv) -> Iterator[ManagerTermBaseCfg]:
    yield from env.reward_manager._term_cfgs
    yield from env.termination_manager._term_cfgs
    for terms in env.event_manager._mode_term_cfgs.values():
        yield from terms
    if isinstance(env.curriculum_manager, CurriculumManager):
        yield from env.curriculum_manager._term_cfgs
    if isinstance(env.metrics_manager, MetricsManager):
        yield from env.metrics_manager._term_cfgs
    for group in env.observation_manager._group_obs_term_cfgs.values():
        yield from group


def validate_state_preservation(env: ManagerBasedRlEnv) -> None:
    """Validate component contracts once, independently of task or term names."""
    for package, audited in (
        ("mjlab", "1.6.0"),
        ("mujoco", "3.8.1"),
        ("mujoco-warp", "3.11.0"),
        ("warp-lang", "1.15.0"),
    ):
        if version(package) != audited:
            raise ValueError(f"Training-env evaluation requires {package}=={audited}")
    if type(env) is not ManagerBasedRlEnv:
        raise TypeError("State preservation requires ManagerBasedRlEnv")
    if env.cfg.auto_reset:
        raise ValueError("State preservation requires adapter-managed reset")
    if env.render_mode is not None or type(env.recorder_manager) is not NullRecorderManager:
        raise ValueError("State preservation requires headless execution without recorders")
    for manager, types in (
        (env.action_manager, (ActionManager,)),
        (env.command_manager, (CommandManager, NullCommandManager)),
        (env.curriculum_manager, (CurriculumManager, NullCurriculumManager)),
        (env.metrics_manager, (MetricsManager, NullMetricsManager)),
        (env.event_manager, (EventManager,)),
        (env.observation_manager, (ObservationManager,)),
        (env.reward_manager, (RewardManager,)),
        (env.termination_manager, (TerminationManager,)),
    ):
        if type(manager) not in types:
            raise TypeError(f"Unsupported mjlab manager: {type(manager).__name__}")
    validate_physics_state(env)
    validate_scene_state(env.scene)
    for name in env.action_manager.active_terms:
        validate_term_state(env.action_manager.get_term(name))
    for name in env.command_manager.active_terms:
        validate_term_state(env.command_manager.get_term(name))
    for cfg in _manager_terms(env):
        validate_term_state(cfg.func)
        if cfg.func is commands_vel and not isinstance(
            env.command_manager.get_term(cfg.params["command_name"]),
            UniformVelocityCommand,
        ):
            raise ValueError("Velocity curricula require a UniformVelocityCommand target")
    observation = env.observation_manager
    for group in observation._group_obs_term_cfgs.values():
        for cfg in group:
            if isinstance(cfg.noise, NoiseCfg):
                validate_term_state(cfg.noise)
    for group in observation._group_obs_class_instances.values():
        for noise in group.values():
            validate_term_state(noise)
    for group in observation._group_obs_term_delay_buffer.values():
        if any(type(buffer) is not DelayBuffer for buffer in group.values()):
            raise TypeError("Unsupported observation delay buffer")
    for group in observation._group_obs_term_history_buffer.values():
        if any(type(buffer) is not CircularBuffer for buffer in group.values()):
            raise TypeError("Unsupported observation history buffer")


@contextmanager
def _preserve_managers(env: ManagerBasedRlEnv) -> Iterator[None]:
    observation = env.observation_manager
    metrics = env.metrics_manager
    tensors = [
        env.episode_length_buf,
        env._manual_reset_pending,
        env._command_dt,
        env.action_manager.action,
        env.action_manager.prev_action,
        env.action_manager.prev_prev_action,
        env.reward_manager._reward_buf,
        env.reward_manager._step_reward,
        *env.reward_manager._episode_sums.values(),
        env.termination_manager._terminated_buf,
        env.termination_manager._truncated_buf,
        *env.termination_manager._term_dones.values(),
        *env.event_manager._interval_term_time_left,
        *env.event_manager._reset_term_last_triggered_step_id,
        *env.event_manager._reset_term_last_triggered_once,
    ]
    substep_count = 0
    if isinstance(metrics, MetricsManager):
        substep_count = metrics._substep_count
        tensors.extend(
            (
                metrics._step_count,
                metrics._step_values,
                *metrics._episode_sums.values(),
                *metrics._episode_max.values(),
                *metrics._substep_accum,
            )
        )
    sim_steps, common_steps = env._sim_step_counter, env.common_step_counter
    obs_buf, obs_cache = env.obs_buf, observation._obs_buffer
    extras = env.extras.copy()
    curriculum_state = env.curriculum_manager._curriculum_state.copy()
    configs = list(_manager_terms(env))
    params = [(cfg, cfg.params.copy()) for cfg in configs]
    weights = [(cfg, cfg.weight) for cfg in env.reward_manager._term_cfgs]
    timeouts = [(cfg, cfg.time_out) for cfg in env.termination_manager._term_cfgs]
    terms: dict[int, object] = {id(cfg.func): cfg.func for cfg in configs}
    for name in env.action_manager.active_terms:
        action = env.action_manager.get_term(name)
        terms[id(action)] = action
    for name in env.command_manager.active_terms:
        command = env.command_manager.get_term(name)
        terms[id(command)] = command
    for group in observation._group_obs_term_cfgs.values():
        for cfg in group:
            if isinstance(cfg.noise, NoiseCfg):
                terms[id(cfg.noise)] = cfg.noise
    for group in observation._group_obs_class_instances.values():
        for noise in group.values():
            terms[id(noise)] = noise
    with ExitStack() as stack:
        stack.enter_context(_preserve_tensors(tensors))
        for term in terms.values():
            stack.enter_context(preserve_term_state(term))
        for group in observation._group_obs_term_history_buffer.values():
            for buffer in group.values():
                stack.enter_context(_preserve_history(buffer))
        for group in observation._group_obs_term_delay_buffer.values():
            for buffer in group.values():
                stack.enter_context(_preserve_delay(buffer))
        try:
            yield
        finally:
            env._sim_step_counter, env.common_step_counter = sim_steps, common_steps
            if isinstance(metrics, MetricsManager):
                metrics._substep_count = substep_count
            env.obs_buf, observation._obs_buffer = obs_buf, obs_cache
            env.extras.clear()
            env.extras.update(extras)
            env.curriculum_manager._curriculum_state.clear()
            env.curriculum_manager._curriculum_state.update(curriculum_state)
            for cfg, saved_params in params:
                cfg.params.clear()
                cfg.params.update(saved_params)
            for cfg, weight in weights:
                cfg.weight = weight
            for cfg, time_out in timeouts:
                cfg.time_out = time_out


@contextmanager
def preserve_mjlab_state(env: ManagerBasedRlEnv) -> Iterator[None]:
    """Preserve a validated environment; topology/config changes require rebuilding it."""
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(env.device) if torch.device(env.device).type == "cuda" else None
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(_preserve_managers(env))
            stack.enter_context(preserve_scene_state(env.scene))
            stack.enter_context(preserve_physics_state(env))
            yield
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, env.device)
