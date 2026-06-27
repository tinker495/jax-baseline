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
from jax_baselines.A2C.impala import IMPALA
from jax_baselines.IMPALA.worker import Impala_Worker
from jax_baselines.PPO.impala_ppo import IMPALA_PPO
from jax_baselines.SPO.impala_spo import IMPALA_SPO
from jax_baselines.TPPO.impala_tppo import IMPALA_TPPO
from replay_memory.replay_factory import make_impala_worker_buffer


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="IMPALA", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="environment")
    parser.add_argument("--worker", type=int, default=16, help="gym_worker_size")
    parser.add_argument("--update_freq", type=int, default=100, help="update frequency")
    parser.add_argument("--algo", type=str, default="A2C", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument("--lamda", type=float, default=1.0, help="lamda")
    parser.add_argument("--rho_max", type=float, default=1.0, help="rho max")
    parser.add_argument("--buffer_size", type=int, default=0, help="buffer size")
    parser.add_argument("--sample_size", type=int, default=1, help="sample_size")
    parser.add_argument("--batch", type=int, default=256, help="batch size")
    parser.add_argument("--steps", type=float, default=1e5, help="step size")
    parser.add_argument("--logdir", type=str, default=default_logdir("impala"), help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="rmsprop", help="optimaizer")
    parser.add_argument("--ent_coef", type=float, default=0.1, help="entropy coefficient")
    parser.add_argument("--val_coef", type=float, default=0.6, help="val coefficient")


def make_workers(args, runtime):
    env_builder, _ = get_env_builder(args.env)
    return [
        runtime.create_worker(Impala_Worker, env_builder, seed=args.seed + i)
        for i in range(args.worker)
    ]


def _common(a):
    return {
        "gamma": a.gamma,
        "lamda": a.lamda,
        "learning_rate": a.learning_rate,
        "update_freq": a.update_freq,
        "batch_size": a.batch,
        "sample_size": a.sample_size,
        "buffer_size": int(a.buffer_size),
        "optimizer_factory": make_batch_scaled_optimizer_factory(a.optimizer, a.batch),
        "val_coef": a.val_coef,
        "ent_coef": a.ent_coef,
        "rho_max": a.rho_max,
        "log_dir": a.logdir,
        "seed": a.seed,
        "worker_replay_factory": make_impala_worker_buffer,
    }


ALGOS = {
    "A2C": AlgoSpec(IMPALA, "ac", _common),
    "PPO": AlgoSpec(IMPALA_PPO, "ac", _common),
    "TPPO": AlgoSpec(IMPALA_TPPO, "ac", _common),
    "SPO": AlgoSpec(IMPALA_SPO, "ac", _common),
}


IMPALA_RUNNER = DistributedFamilyRunner(
    add_args=add_args,
    make_workers=make_workers,
    policy_kwargs=default_policy_kwargs,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.ac",
    variant=lambda _args: "",
    ray_cpu_headroom=4,
)


def main(argv=None):
    run_distributed_family(IMPALA_RUNNER, argv)
    return 0


if __name__ == "__main__":
    main()
