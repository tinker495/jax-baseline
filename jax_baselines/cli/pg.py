from jax_baselines.A2C.a2c import A2C
from jax_baselines.cli._common import default_logdir, set_default_xla_flags
from jax_baselines.cli._run import AlgoSpec, FamilyRunner, run_family
from jax_baselines.common.env_builder import get_env_builder
from jax_baselines.PPO.ppo import PPO
from jax_baselines.SPO.spo import SPO
from jax_baselines.TPPO.tppo import TPPO

set_default_xla_flags()


def add_args(parser):
    parser.add_argument("--experiment_name", type=str, default="PG", help="experiment name")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="environment")
    parser.add_argument(
        "--env_backend",
        type=str,
        default="gymnasium",
        choices=["gymnasium", "envpool"],
        help="vectorized-env backend when worker>1 (gymnasium default; envpool is faster)",
    )
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="A2C", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument("--lamda", type=float, default=0.95, help="gae lamda")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--mini_batch", type=int, default=32, help="batch size")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--logdir", type=str, default=default_logdir("pg"), help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="entropy coefficient")
    parser.add_argument("--val_coef", type=float, default=0.6, help="val coefficient")
    parser.add_argument("--epoch_num", type=int, default=4, help="epoch number")
    parser.add_argument("--gae_normalize", action="store_true")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument("--use_entropy_adv_shaping", action="store_true")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    parser.set_defaults(gae_normalize=False)


def build_env(args):
    env_builder, _ = get_env_builder(
        args.env,
        env_backend=args.env_backend,
        timescale=args.time_scale,
        capture_frame_rate=args.capture_frame_rate,
    )
    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "embedding_mode": "normal"}
    return env_builder, policy_kwargs


def _common(a):
    return {
        "num_workers": a.worker,
        "gamma": a.gamma,
        "batch_size": a.batch,
        "val_coef": a.val_coef,
        "ent_coef": a.ent_coef,
        "use_entropy_adv_shaping": a.use_entropy_adv_shaping,
        "log_dir": a.logdir,
        "optimizer": a.optimizer,
        "seed": a.seed,
    }


def _ppo_like(a):
    return {
        **_common(a),
        "lamda": a.lamda,
        "gae_normalize": a.gae_normalize,
        "minibatch_size": a.mini_batch,
        "epoch_num": a.epoch_num,
    }


ALGOS = {
    "A2C": AlgoSpec(A2C, "ac", _common),
    "PPO": AlgoSpec(PPO, "ac", _ppo_like),
    "TPPO": AlgoSpec(TPPO, "ac", _ppo_like),
    "SPO": AlgoSpec(SPO, "ac", _ppo_like),
}


PG_RUNNER = FamilyRunner(
    add_args=add_args,
    build_env=build_env,
    algos=ALGOS,
    maker_pkg="model_builder.{lib}.ac",
    variant=lambda _args: "",
)


def main(argv=None):
    return run_family(PG_RUNNER, argv)


if __name__ == "__main__":
    main()
