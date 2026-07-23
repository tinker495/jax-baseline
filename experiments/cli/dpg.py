# isort: off
from experiments.cli._common import default_logdir
from env_builder.env_builder import get_env_builder

# isort: on
from experiments.cli._run import AlgoSpec, FamilyRunner, run_family
from experiments.optimizers import make_batch_scaled_optimizer_factory
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.SAC.sac import SAC
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TQC.tqc import TQC
from jax_baselines.XQC.xqc import XQC
from replay_memory.replay_factory import make_replay_buffer


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="DPG", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="environment")
    parser.add_argument(
        "--env_backend",
        type=str,
        default="gymnasium",
        choices=["gymnasium", "envpool"],
        help="vectorized-env backend when worker>1 (gymnasium default; envpool is faster)",
    )
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="DDPG", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument(
        "--target_update_tau", type=float, default=2e-3, help="target update intervals"
    )
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=100000, help="buffer_size")
    parser.add_argument("--per", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--scaled_by_reset", action="store_true")
    simba_group = parser.add_mutually_exclusive_group()
    simba_group.add_argument("--simba", action="store_true")
    simba_group.add_argument("--simbav2", action="store_true")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument(
        "--eval_num",
        type=int,
        default=100,
        help="number of evaluations over the whole training run (default 100)",
    )
    parser.add_argument("--logdir", type=str, default=default_logdir("dpg"), help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_support", type=int, default=25, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--mixture", type=str, default="truncated", help="mixture type")
    parser.add_argument("--quantile_drop", type=float, default=0.1, help="quantile_drop ratio")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--action_noise", type=float, default=0.1, help="action_noise")
    parser.add_argument("--optimizer", type=str, default="adopt", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient_steps")
    parser.add_argument("--max_bulk_updates_per_pulse", type=int, default=32)
    parser.add_argument("--train_freq", type=int, default=1, help="train_frequancy")
    parser.add_argument("--critic_num", type=int, default=2, help="tqc critic number")
    parser.add_argument(
        "--ent_coef",
        type=str,
        default=None,
        help="entropy coefficient (algorithm default if omitted)",
    )
    parser.add_argument("--sigma_target", type=float, default=0.15)
    parser.add_argument("--actor_update_period", type=int, default=2)
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--use_checkpointing", action="store_true")
    reward_group = parser.add_mutually_exclusive_group()
    reward_group.add_argument(
        "--reward_normalization",
        dest="reward_normalization",
        action="store_true",
        help="normalize sampled rewards by discounted-return RMS",
    )
    reward_group.add_argument(
        "--no_reward_normalization",
        dest="reward_normalization",
        action="store_false",
        help="disable discounted-return reward normalization",
    )
    parser.set_defaults(reward_normalization=None)


def build_env(args):
    env_builder, _ = get_env_builder(
        args.env,
        env_backend=args.env_backend,
    )
    policy_kwargs = {
        "node": args.node,
        "hidden_n": args.hidden_n,
        "embedding_mode": "normal",
    }
    return env_builder, policy_kwargs


def _variant(args):
    return "simbav2_" if args.simbav2 else "simba_" if args.simba else ""


def _simba(args):
    return args.simba or args.simbav2


def _common(a):
    return {
        "num_workers": a.worker,
        "gamma": a.gamma,
        "learning_rate": a.learning_rate,
        "batch_size": a.batch,
        "buffer_size": int(a.buffer_size),
        "learning_starts": a.learning_starts,
        "prioritized_replay": a.per,
        "scaled_by_reset": a.scaled_by_reset,
        "simba": _simba(a),
        "simba_v2": a.simbav2,
        "n_step": a.n_step,
        "train_freq": a.train_freq,
        "seed": a.seed,
        "gradient_steps": a.gradient_steps,
        "max_bulk_updates_per_pulse": a.max_bulk_updates_per_pulse,
        "log_dir": a.logdir,
        "optimizer_factory": make_batch_scaled_optimizer_factory(a.optimizer, a.batch),
        "replay_factory": make_replay_buffer,
        "use_checkpointing": a.use_checkpointing,
        "reward_normalization": (
            a.algo == "XQC" if a.reward_normalization is None else a.reward_normalization
        ),
    }


ALGOS = {
    "DDPG": AlgoSpec(
        DDPG,
        "ddpg",
        lambda a: {**_common(a), "target_network_update_tau": a.target_update_tau},
    ),
    "TD3": AlgoSpec(
        TD3,
        "td3",
        lambda a: {
            **_common(a),
            "target_network_update_tau": a.target_update_tau,
            "action_noise": a.action_noise,
        },
    ),
    "SAC": AlgoSpec(
        SAC,
        "sac",
        lambda a: {
            **_common(a),
            "target_network_update_tau": a.target_update_tau,
            "ent_coef": a.ent_coef if a.ent_coef is not None else "auto_0.01",
            "sigma_target": a.sigma_target,
            "actor_update_period": a.actor_update_period,
        },
    ),
    "CrossQ": AlgoSpec(
        CrossQ,
        "crossq",
        lambda a: {
            **_common(a),
            "ent_coef": a.ent_coef if a.ent_coef is not None else "auto",
            "sigma_target": a.sigma_target,
        },
    ),
    "XQC": AlgoSpec(
        XQC,
        "xqc",
        lambda a: {
            **_common(a),
            "target_network_update_tau": a.target_update_tau,
            "ent_coef": a.ent_coef if a.ent_coef is not None else "auto_0.01",
            "sigma_target": a.sigma_target,
        },
    ),
    "TQC": AlgoSpec(
        TQC,
        "tqc",
        lambda a: {
            **_common(a),
            "target_network_update_tau": a.target_update_tau,
            "ent_coef": a.ent_coef if a.ent_coef is not None else "auto",
            "quantile_drop": a.quantile_drop,
            "n_support": a.n_support,
            "critic_num": a.critic_num,
            "mixture_type": a.mixture,
        },
    ),
    # TD7 intentionally does NOT reuse _common(): TD7.__init__ forces
    # n_step=1, target_network_update_tau=0, prioritized_replay=True and
    # use_checkpointing=True via td7_kwargs, where caller kwargs win over those
    # defaults. Passing the CLI defaults that _common() carries would silently
    # override TD7's required settings, so this dict must omit those keys.
    "TD7": AlgoSpec(
        TD7,
        "td7",
        lambda a: {
            "num_workers": a.worker,
            "gamma": a.gamma,
            "learning_rate": a.learning_rate,
            "batch_size": a.batch,
            "buffer_size": int(a.buffer_size),
            "target_network_update_freq": 250,
            "learning_starts": a.learning_starts,
            "action_noise": a.action_noise,
            "train_freq": a.train_freq,
            "scaled_by_reset": a.scaled_by_reset,
            "simba": _simba(a),
            "simba_v2": a.simbav2,
            "seed": a.seed,
            "gradient_steps": a.gradient_steps,
            "max_bulk_updates_per_pulse": a.max_bulk_updates_per_pulse,
            "log_dir": a.logdir,
            "optimizer_factory": make_batch_scaled_optimizer_factory(a.optimizer, a.batch),
            "replay_factory": make_replay_buffer,
            "reward_normalization": (
                False if a.reward_normalization is None else a.reward_normalization
            ),
        },
    ),
}


DPG_RUNNER = FamilyRunner(
    add_args=add_args,
    build_env=build_env,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.dpg",
    variant=_variant,
)


def main(argv=None):
    run_family(DPG_RUNNER, argv)
    return 0


if __name__ == "__main__":
    main()
