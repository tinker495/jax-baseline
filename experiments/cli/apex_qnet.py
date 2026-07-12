# isort: off
from experiments.cli._common import default_logdir
from env_builder.env_builder import get_env_builder

# isort: on
from experiments.cli._run import (
    AlgoSpec,
    DistributedFamilyRunner,
    default_policy_kwargs,
    run_distributed_family,
)
from experiments.optimizers import make_batch_scaled_optimizer_factory
from jax_baselines.APE_X.worker import Ape_X_Worker
from jax_baselines.C51.apex_c51 import APE_X_C51
from jax_baselines.DQN.apex_dqn import APE_X_DQN
from jax_baselines.IQN.apex_iqn import APE_X_IQN
from jax_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN
from replay_memory.replay_factory import make_apex_replay


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="APEX", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment")
    parser.add_argument("--algo", type=str, default="DQN", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--target_update", type=int, default=2000, help="target update intervals")
    parser.add_argument("--batch_num", type=int, default=16, help="batch size")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=200000, help="buffer_size")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--noisynet", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--munchausen", action="store_true")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument(
        "--logdir", type=str, default=default_logdir("apex_qnet"), help="log file dir"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max", type=float, default=250, help="c51 max")
    parser.add_argument("--min", type=float, default=-250, help="c51 min")
    parser.add_argument("--n_support", type=int, default=200, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--delta", type=float, default=0.001, help="delta for QRDQN,IQN,FQF")
    parser.add_argument("--CVaR", type=float, default=1.0, help="IQN risk avoiding factor")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient steps")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--initial_eps", type=float, default=0.4, help="initial epsilon")
    parser.add_argument("--eps_decay", type=float, default=3, help="exploration fraction")
    parser.add_argument("--compress_memory", action="store_true")


def make_workers(args, runtime):
    env_builder, _ = get_env_builder(args.env)
    return [
        runtime.create_worker(Ape_X_Worker, env_builder, seed=args.seed + i)
        for i in range(args.worker)
    ]


def _common(a):
    return {
        "gamma": a.gamma,
        "learning_rate": a.learning_rate,
        "batch_num": a.batch_num,
        "mini_batch_size": a.batch_size,
        "buffer_size": int(a.buffer_size),
        "target_network_update_freq": a.target_update,
        "double_q": a.double,
        "dueling_model": a.dueling,
        "exploration_initial_eps": a.initial_eps,
        "exploration_decay": a.eps_decay,
        "param_noise": a.noisynet,
        "n_step": a.n_step,
        "munchausen": a.munchausen,
        "gradient_steps": a.gradient_steps,
        "learning_starts": a.learning_starts,
        "log_dir": a.logdir,
        "optimizer_factory": make_batch_scaled_optimizer_factory(
            a.optimizer, a.batch_num * a.batch_size
        ),
        "compress_memory": a.compress_memory,
        "seed": a.seed,
        "apex_replay_factory": make_apex_replay,
    }


ALGOS = {
    "DQN": AlgoSpec(APE_X_DQN, "dqn", _common),
    "C51": AlgoSpec(
        APE_X_C51,
        "c51",
        lambda a: {**_common(a), "categorial_max": a.max, "categorial_min": a.min},
    ),
    "QRDQN": AlgoSpec(
        APE_X_QRDQN,
        "qrdqn",
        lambda a: {**_common(a), "n_support": a.n_support, "delta": a.delta},
    ),
    "IQN": AlgoSpec(
        APE_X_IQN,
        "iqn",
        lambda a: {
            **_common(a),
            "n_support": a.n_support,
            "delta": a.delta,
            "CVaR": a.CVaR,
        },
    ),
}


APEX_QNET_RUNNER = DistributedFamilyRunner(
    add_args=add_args,
    make_workers=make_workers,
    policy_kwargs=default_policy_kwargs,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.qnet",
    variant=lambda _args: "",
    ray_cpu_headroom=2,
)


def main(argv=None):
    run_distributed_family(APEX_QNET_RUNNER, argv)
    return 0


if __name__ == "__main__":
    main()
