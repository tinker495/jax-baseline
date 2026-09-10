"""Native mjlab integration state and randomized model buffers, restored in place."""

from collections.abc import Iterator
from contextlib import contextmanager

import mujoco_warp as mjwarp
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from env_builder.mjlab_scene_state import _preserve_tensors


def _model_arrays(model: mjwarp.Model) -> dict[str, wp.array]:
    # mjlab's declared DR fields plus MuJoCo's derived model constants.
    return {
        "actuator_acc0": model.actuator_acc0,
        "actuator_biasprm": model.actuator_biasprm,
        "actuator_forcerange": model.actuator_forcerange,
        "actuator_gainprm": model.actuator_gainprm,
        "body_inertia": model.body_inertia,
        "body_invweight0": model.body_invweight0,
        "body_ipos": model.body_ipos,
        "body_iquat": model.body_iquat,
        "body_mass": model.body_mass,
        "body_pos": model.body_pos,
        "body_quat": model.body_quat,
        "body_subtreemass": model.body_subtreemass,
        "cam_fovy": model.cam_fovy,
        "cam_intrinsic": model.cam_intrinsic,
        "cam_mat0": model.cam_mat0,
        "cam_pos": model.cam_pos,
        "cam_pos0": model.cam_pos0,
        "cam_poscom0": model.cam_poscom0,
        "cam_quat": model.cam_quat,
        "dof_armature": model.dof_armature,
        "dof_damping": model.dof_damping,
        "dof_frictionloss": model.dof_frictionloss,
        "dof_invweight0": model.dof_invweight0,
        "eq_data": model.eq_data,
        "geom_aabb": model.geom_aabb,
        "geom_dataid": model.geom_dataid,
        "geom_friction": model.geom_friction,
        "geom_matid": model.geom_matid,
        "geom_pos": model.geom_pos,
        "geom_quat": model.geom_quat,
        "geom_rbound": model.geom_rbound,
        "geom_rgba": model.geom_rgba,
        "geom_size": model.geom_size,
        "jnt_actfrcrange": model.jnt_actfrcrange,
        "jnt_range": model.jnt_range,
        "jnt_stiffness": model.jnt_stiffness,
        "light_ambient": model.light_ambient,
        "light_attenuation": model.light_attenuation,
        "light_cutoff": model.light_cutoff,
        "light_diffuse": model.light_diffuse,
        "light_dir": model.light_dir,
        "light_dir0": model.light_dir0,
        "light_exponent": model.light_exponent,
        "light_pos": model.light_pos,
        "light_pos0": model.light_pos0,
        "light_poscom0": model.light_poscom0,
        "light_specular": model.light_specular,
        "mat_emission": model.mat_emission,
        "mat_rgba": model.mat_rgba,
        "mat_shininess": model.mat_shininess,
        "mat_specular": model.mat_specular,
        "mat_texid": model.mat_texid,
        "mat_texrepeat": model.mat_texrepeat,
        "pair_friction": model.pair_friction,
        "qpos0": model.qpos0,
        "qpos_spring": model.qpos_spring,
        "site_pos": model.site_pos,
        "site_quat": model.site_quat,
        "stat.meaninertia": model.stat.meaninertia,
        "tendon_actfrcrange": model.tendon_actfrcrange,
        "tendon_armature": model.tendon_armature,
        "tendon_damping": model.tendon_damping,
        "tendon_frictionloss": model.tendon_frictionloss,
        "tendon_invweight0": model.tendon_invweight0,
        "tendon_length0": model.tendon_length0,
        "tendon_lengthspring": model.tendon_lengthspring,
        "tendon_stiffness": model.tendon_stiffness,
    }


def validate_physics_state(env: ManagerBasedRlEnv) -> None:
    if env.sim.nan_guard.enabled:
        raise ValueError("State preservation requires NaN recording to be disabled")
    if env.sim.mj_model.nplugin:
        raise ValueError("State preservation does not support native plugin-owned state")
    unsupported = env.sim.expanded_fields - _model_arrays(env.sim.wp_model).keys()
    if unsupported:
        raise ValueError(f"Unsupported randomized model fields: {sorted(unsupported)}")


@contextmanager
def preserve_physics_state(env: ManagerBasedRlEnv) -> Iterator[None]:
    sim = env.sim
    model, data = sim.wp_model, sim.wp_data
    default_fields = sim.default_model_fields.copy()
    with wp.ScopedDevice(sim.wp_device):
        # MuJoCo 3.8 INTEGRATION layout, including warmstart and history.
        state = wp.array2d(
            shape=(
                env.num_envs,
                1
                + model.nq
                + 3 * model.nv
                + model.na
                + model.nu
                + 6 * model.nbody
                + model.neq
                + 7 * model.nmocap
                + model.nuserdata
                + model.nhistory,
            ),
            dtype=float,
        )
        mjwarp.get_state(model, data, state, int(mjwarp.State.INTEGRATION))
        model_state = [(array, wp.clone(array)) for array in _model_arrays(model).values()]
        caches = [
            (array, wp.clone(array))
            for array in (
                data.qacc,
                data.actuator_force,
                data.qfrc_actuator,
                data.sensordata,
                sim._reset_mask_wp,
            )
        ]
        sleep_state = [
            (array, wp.clone(array))
            for array in (
                data.tree_asleep,
                data.tree_awake,
                data.body_awake,
                data.body_awake_ind,
                data.dof_awake_ind,
                data.ntree_awake,
                data.nbody_awake,
                data.nv_awake,
            )
        ]
    with _preserve_tensors(tuple(default_fields.values())):
        try:
            yield
        finally:
            with wp.ScopedDevice(sim.wp_device):
                for destination, saved in (*model_state, *sleep_state):
                    wp.copy(destination, saved)
                mjwarp.set_state(model, data, state, int(mjwarp.State.INTEGRATION))
                sim.forward()
                # Forward reconstructs derived poses and overwrites warmstart/caches.
                mjwarp.set_state(model, data, state, int(mjwarp.State.INTEGRATION))
                for destination, saved in (*sleep_state, *caches):
                    wp.copy(destination, saved)
            sim.default_model_fields.clear()
            sim.default_model_fields.update(default_fields)
