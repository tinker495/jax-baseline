"""Shared ``learn()`` lifecycle for the training families.

The session sits one seam *above* the rollout loop. Where
:mod:`jax_baselines.core.rollout` owns the per-step environment loop, the
session owns the run *lifecycle* that wrapped it: run-name tagging, schedule
setup, ``eval_freq``, the ``pbar``, the logger lifecycle, and the closing
eval + save.

The session is **loop-agnostic** — :meth:`TrainingSession.run` never touches a
:class:`~jax_baselines.core.rollout.RolloutEngine`; it only calls
``agent.run_training_loop(ctx)``. The coupling to the rollout engine lives
solely in :func:`off_policy_loop`, the off-policy rollout dispatch that the
Q-Net and DPG families delegate to from their ``run_training_loop``. That keeps
the door open for the on-policy A2C family to reuse the same session through its
own (non-``RolloutEngine``) ``run_training_loop`` later.

Per-run state travels to the agent explicitly in a :class:`RunContext`, never
read back off ``self``. ``ctx.logger_run`` is valid only inside the session's
``with logger`` block; the agent must not retain ``ctx`` past ``run()``.
"""

from dataclasses import dataclass

from jax_baselines.core.rollout import RolloutEngine
from jax_baselines.core.runtime_adapters import NoOpLogger, make_progress


@dataclass(frozen=True)
class RunContext:
    """Per-run state the session threads to the agent as an argument.

    ``logger_run`` is valid only inside the session's ``with logger`` block.
    """

    logger_run: object
    eval_freq: int
    pbar: object
    log_interval: int


def eval_freq_from_count(eval_num, total_timesteps, worker_size):
    """Convert a target number of evaluations over the whole run into an eval
    cadence in env steps, snapped to a ``worker_size`` multiple so the
    vectorized step counter lands on it and floored at ``worker_size``.

    The default of 100 evals reproduces the historical ``total_timesteps // 100``
    cadence.
    """
    eval_num = max(1, int(eval_num))
    step = total_timesteps // eval_num
    return max(worker_size, (step // worker_size) * worker_size)


class TrainingSession:
    """Owns the ``learn()`` lifecycle shared across the training families."""

    def run(
        self,
        agent,
        total_timesteps,
        callback,
        log_interval,
        experiment_name,
        run_name,
        eval_num=100,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    ):
        run_name = agent.run_name_update(run_name)
        agent.prepare_run(total_timesteps)
        eval_freq = eval_freq_from_count(eval_num, total_timesteps, agent.worker_size)
        logger_factory = logger_factory or NoOpLogger
        progress_factory = progress_factory or make_progress
        pbar = progress_factory(0, total_timesteps, agent.worker_size, miniters=log_interval)
        logger = logger_factory(run_name, experiment_name, agent.log_dir, agent)
        # ``test()`` is a separate entry point that re-enters the run's logger, so the
        # agent keeps a reference to it (matching the pre-refactor learn() contract).
        agent.logger = logger
        if record_test_fn is not None:
            agent.record_test_fn = record_test_fn
        elif hasattr(agent, "record_test_fn"):
            delattr(agent, "record_test_fn")
        try:
            with logger as logger_run:
                ctx = RunContext(logger_run, eval_freq, pbar, log_interval)
                agent.run_training_loop(ctx)
                agent.eval(ctx, total_timesteps)
                agent.save_params(logger_run.get_local_path("params"))
        finally:
            release_run_context = getattr(agent, "release_run_context", None)
            if release_run_context is not None:
                release_run_context()


def off_policy_loop(agent, ctx):
    """Off-policy rollout dispatch: the seam between the session and the engine.

    Owns the 4-way ``env_type`` x ``use_checkpointing`` dispatch for the Q-Net
    and DPG families; each family's ``run_training_loop`` delegates here.
    """
    engine = RolloutEngine(agent.make_rollout_spec(ctx))
    single = agent.env_type == "SingleEnv"
    ckpt = agent.use_checkpointing
    if single and ckpt:
        engine.learn_single_env_checkpointing(ctx.pbar, log_interval=ctx.log_interval)
    elif single:
        engine.learn_single_env(ctx.pbar, log_interval=ctx.log_interval)
    elif ckpt:
        engine.learn_vectorized_env_checkpointing(ctx.pbar, log_interval=ctx.log_interval)
    else:
        engine.learn_vectorized_env(ctx.pbar, log_interval=ctx.log_interval)
