"""Serializable checkpoint state for the off-policy DPG family.

Sibling to :mod:`jax_baselines.common.checkpoint` (the schedule) and
:mod:`jax_baselines.common.rollout` (the rollout loop): where
:class:`~jax_baselines.common.checkpoint.CheckpointController` owns the checkpoint
*schedule*, this module owns the checkpoint *state* an agent persists to resume
training and run eval-consistent behaviour.

:class:`CheckpointState` is the family-wide spine. The per-algorithm network
bundle -- the only part that varies across DDPG/TD3/SAC/TQC/CrossQ/TD7 -- is
supplied by each algorithm as its own typed ``flax.struct`` bundle and carried in
the ``params`` field (with the eval-time snapshot in ``eval_snapshot``). The spine
itself never names a concrete algorithm's parameters.

Serialization rides the existing pytree
:func:`~jax_baselines.common.serialization.save` / ``restore``: every field is a
pytree node, so the struct *is* the wire format. There is no spine-level
``to_state``/``from_state`` ladder; the sibling ``CheckpointController`` and
``RunningMeanStd`` keep their own and are embedded here as plain ``dict`` fields.
"""

from typing import Any, Optional

from flax import struct


@struct.dataclass
class CheckpointState:
    """The serializable spine an off-policy DPG agent saves and restores.

    Fields:
        params: The algorithm's own ``flax.struct`` checkpoint bundle (the real
            seam). The spine treats it opaquely.
        train_steps_count: Gradient-step counter (``np.int64`` scalar).
        ckpt_residual: Training-cadence residual carried across save/load
            (``np.float32`` scalar).
        controller_state: ``CheckpointController.to_state()`` schedule snapshot.
        eval_snapshot: The ``{"encoder", "policy"}`` behaviour-state captured for
            eval-consistent action selection; ``None`` until the first snapshot.
        obs_rms_state / action_obs_rms_state / checkpoint_obs_rms_state: simba
            observation-normalizer states (``RunningMeanStd.to_state()``); ``None``
            when simba is disabled or the normalizer has not been snapshotted.
    """

    params: Any
    train_steps_count: Any
    ckpt_residual: Any
    controller_state: dict
    eval_snapshot: Optional[Any] = None
    obs_rms_state: Optional[dict] = None
    action_obs_rms_state: Optional[dict] = None
    checkpoint_obs_rms_state: Optional[dict] = None
