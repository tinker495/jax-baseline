from env_builder.env_builder import get_env_builder
from experiments.cli._common import default_logdir, set_default_xla_flags
from experiments.cli._run import (
    AlgoSpec,
    FamilyRunner,
    default_replay_factory,
    run_family,
)
from experiments.optimizers import (
    make_batch_scaled_optimizer_factory,
    make_fqf_optimizer_factory,
    make_optimizer_factory,
)
from jax_baselines.BBF.bbf import BBF
from jax_baselines.BBF.hl_gauss_bbf import HL_GAUSS_BBF
from jax_baselines.C51.c51 import C51
from jax_baselines.C51.hl_gauss_c51 import HL_GAUSS_C51
from jax_baselines.DQN.dqn import DQN
from jax_baselines.FQF.fqf import FQF
from jax_baselines.IQN.iqn import IQN
from jax_baselines.QRDQN.qrdqn import QRDQN
from jax_baselines.SPR.hl_gauss_spr import HL_GAUSS_SPR
from jax_baselines.SPR.spr import SPR

set_default_xla_flags()


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="Q_network", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment")
    parser.add_argument(
        "--env_backend",
        type=str,
        default="gymnasium",
        choices=["gymnasium", "envpool"],
        help="vectorized-env backend when worker>1 (gymnasium default; envpool is faster)",
    )
    parser.add_argument("--algo", type=str, default="DQN", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--target_update", type=int, default=2000, help="target update intervals")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=200000, help="buffer_size")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--noisynet", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--off_policy_fix", action="store_true")
    parser.add_argument("--munchausen", action="store_true")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument(
        "--eval_num",
        type=int,
        default=100,
        help="number of evaluations over the whole training run (default 100)",
    )
    parser.add_argument("--logdir", type=str, default=default_logdir("qnet"), help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max", type=float, default=10, help="c51 max")
    parser.add_argument("--min", type=float, default=-10, help="c51 min")
    parser.add_argument("--n_support", type=int, default=32, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="huber loss delta  for QRDQN,IQN,FQF"
    )
    parser.add_argument("--CVaR", type=float, default=1.0, help="IQN risk avoiding factor")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--final_eps", type=float, default=0.1, help="final epsilon")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--train_freq", type=int, default=1, help="train_frequancy")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient steps")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument(
        "--exploration_fraction", type=float, default=0.3, help="exploration fraction"
    )
    parser.add_argument("--compress_memory", action="store_true")
    parser.add_argument("--hl_gauss", action="store_true")
    parser.add_argument("--scaled_by_reset", action="store_true")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    parser.add_argument("--use_checkpointing", action="store_true")


def build_env(args):
    env_builder, _ = get_env_builder(
        args.env,
        env_backend=args.env_backend,
        timescale=args.time_scale,
        capture_frame_rate=args.capture_frame_rate,
    )
    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n}
    return env_builder, policy_kwargs


def _common(a):
    """Kwargs shared by the standard value-based algos (DQN/C51/QRDQN/IQN/FQF)."""
    return {
        "num_workers": a.worker,
        "seed": a.seed,
        "gamma": a.gamma,
        "learning_rate": a.learning_rate,
        "batch_size": a.batch,
        "buffer_size": int(a.buffer_size),
        "target_network_update_freq": a.target_update,
        "prioritized_replay": a.per,
        "double_q": a.double,
        "dueling_model": a.dueling,
        "exploration_final_eps": a.final_eps,
        "param_noise": a.noisynet,
        "n_step": a.n_step,
        "munchausen": a.munchausen,
        "gradient_steps": a.gradient_steps,
        "train_freq": a.train_freq,
        "learning_starts": a.learning_starts,
        "exploration_fraction": a.exploration_fraction,
        "log_dir": a.logdir,
        "optimizer_factory": make_batch_scaled_optimizer_factory(a.optimizer, a.batch),
        "compress_memory": a.compress_memory,
        "replay_factory": default_replay_factory(),
        "use_checkpointing": a.use_checkpointing,
    }


def _bbf_optimizer_factory(a):
    """Preserve BBF-family optimizer policy after moving selection to experiments.

    Plain BBF historically honored the CLI optimizer name, while HL_GAUSS_BBF
    selected AdamW with weight_decay=0.1 inside setup_model. Keep that
    algorithm-specific policy in the experiments adapter rather than the core.
    """
    if a.hl_gauss:
        return make_optimizer_factory("adamw", eps=1e-8, weight_decay=0.1)
    return make_optimizer_factory(a.optimizer, weight_decay=0.1)


ALGOS = {
    "DQN": AlgoSpec(DQN, "dqn", _common),
    "C51": AlgoSpec(
        lambda a: HL_GAUSS_C51 if a.hl_gauss else C51,
        "c51",
        lambda a: {**_common(a), "categorial_max": a.max, "categorial_min": a.min},
    ),
    "QRDQN": AlgoSpec(
        QRDQN,
        "qrdqn",
        lambda a: {**_common(a), "delta": a.delta, "n_support": a.n_support},
    ),
    "IQN": AlgoSpec(
        IQN,
        "iqn",
        lambda a: {
            **_common(a),
            "delta": a.delta,
            "n_support": a.n_support,
            "CVaR": a.CVaR,
        },
    ),
    "FQF": AlgoSpec(
        FQF,
        "fqf",
        lambda a: {
            **_common(a),
            "delta": a.delta,
            "n_support": a.n_support,
            "fqf_optimizer_factory": make_fqf_optimizer_factory(),
        },
    ),
    # SPR forces a self-prediction loop: its old construction block carries a
    # reduced kwarg set (off_policy_fix/scaled_by_reset + categorial bounds, no
    # target_update/per/double/dueling/exploration/n_step/train_freq), so it does
    # NOT reuse _common().
    "SPR": AlgoSpec(
        lambda a: HL_GAUSS_SPR if a.hl_gauss else SPR,
        "spr",
        lambda a: {
            "num_workers": a.worker,
            "seed": a.seed,
            "gamma": a.gamma,
            "learning_rate": a.learning_rate,
            "batch_size": a.batch,
            "buffer_size": int(a.buffer_size),
            "off_policy_fix": a.off_policy_fix,
            "scaled_by_reset": a.scaled_by_reset,
            "munchausen": a.munchausen,
            "gradient_steps": a.gradient_steps,
            "learning_starts": a.learning_starts,
            "categorial_max": a.max,
            "categorial_min": a.min,
            "log_dir": a.logdir,
            "optimizer_factory": make_batch_scaled_optimizer_factory(a.optimizer, a.batch),
            "compress_memory": a.compress_memory,
            "replay_factory": default_replay_factory(),
            "use_checkpointing": a.use_checkpointing,
        },
    ),
    # BBF likewise carries its own reduced set (exploration + off_policy_fix +
    # categorial bounds, no target_update/per/double/dueling/n_step/scaled_by_reset).
    "BBF": AlgoSpec(
        lambda a: HL_GAUSS_BBF if a.hl_gauss else BBF,
        "bbf",
        lambda a: {
            "num_workers": a.worker,
            "seed": a.seed,
            "gamma": a.gamma,
            "learning_rate": a.learning_rate,
            "batch_size": a.batch,
            "buffer_size": int(a.buffer_size),
            "exploration_final_eps": a.final_eps,
            "param_noise": a.noisynet,
            "off_policy_fix": a.off_policy_fix,
            "munchausen": a.munchausen,
            "gradient_steps": a.gradient_steps,
            "train_freq": a.train_freq,
            "learning_starts": a.learning_starts,
            "categorial_max": a.max,
            "categorial_min": a.min,
            "exploration_fraction": a.exploration_fraction,
            "log_dir": a.logdir,
            "optimizer_factory": _bbf_optimizer_factory(a),
            "compress_memory": a.compress_memory,
            "replay_factory": default_replay_factory(),
            "use_checkpointing": a.use_checkpointing,
        },
    ),
}


QNET_RUNNER = FamilyRunner(
    add_args=add_args,
    build_env=build_env,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.qnet",
    variant=lambda _args: "",
)


def main(argv=None):
    return run_family(QNET_RUNNER, argv)


if __name__ == "__main__":
    main()
