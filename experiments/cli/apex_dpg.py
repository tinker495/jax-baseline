from env_builder.env_builder import get_env_builder
from experiments.cli._common import default_logdir
from experiments.cli._run import (
    AlgoSpec,
    DistributedFamilyRunner,
    default_policy_kwargs,
    run_distributed_family,
)
from experiments.optimizers import make_batch_scaled_optimizer_factory
from jax_baselines.APE_X.dpg_worker import Ape_X_Worker
from jax_baselines.DDPG.apex_ddpg import APE_X_DDPG
from jax_baselines.TD3.apex_td3 import APE_X_TD3
from replay_memory.replay_factory import (
    make_multi_prioritized_buffer,
    make_worker_replay_buffer,
)


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="APEX_DPG", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="environment")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="DDPG", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument(
        "--target_update_tau", type=float, default=2e-3, help="target update intervals"
    )
    parser.add_argument("--batch_num", type=int, default=16, help="batch num")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=100000, help="buffer_size")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument(
        "--logdir", type=str, default=default_logdir("apex_dpg"), help="log file dir"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="adopt", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient_steps")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--initial_eps", type=float, default=0.4, help="initial epsilon")
    parser.add_argument("--eps_decay", type=float, default=3, help="exploration fraction")


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
        "target_network_update_tau": a.target_update_tau,
        "learning_starts": a.learning_starts,
        "exploration_initial_eps": a.initial_eps,
        "exploration_decay": a.eps_decay,
        "n_step": a.n_step,
        "seed": a.seed,
        "gradient_steps": a.gradient_steps,
        "log_dir": a.logdir,
        "optimizer_factory": make_batch_scaled_optimizer_factory(
            a.optimizer, a.batch_num * a.batch_size
        ),
        "multi_replay_factory": make_multi_prioritized_buffer,
        "worker_replay_factory": make_worker_replay_buffer,
    }


ALGOS = {
    "DDPG": AlgoSpec(APE_X_DDPG, "ddpg", _common),
    "TD3": AlgoSpec(APE_X_TD3, "td3", _common),
}


APEX_DPG_RUNNER = DistributedFamilyRunner(
    add_args=add_args,
    make_workers=make_workers,
    policy_kwargs=default_policy_kwargs,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.dpg",
    variant=lambda _args: "",
    ray_cpu_headroom=2,
)


def main(argv=None):
    run_distributed_family(APEX_DPG_RUNNER, argv)
    return 0


if __name__ == "__main__":
    main()
