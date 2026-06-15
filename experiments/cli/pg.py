from experiments.cli._common import default_logdir, set_default_xla_flags
from experiments.cli._run import AlgoSpec, FamilyRunner, run_family
from jax_baselines.A2C.a2c import A2C
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
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument("--lamda", type=float, default=0.95, help="gae lamda")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--mini_batch", type=int, default=32, help="batch size")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument(
        "--eval_num",
        type=int,
        default=100,
        help="number of evaluations over the whole training run (default 100)",
    )
    parser.add_argument("--logdir", type=str, default=default_logdir("pg"), help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=1e-2 / 256.0,
        help="optimizer epsilon for Adam-like optimizers",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="global gradient norm clipping threshold; omitted disables clipping",
    )
    parser.add_argument(
        "--lr_annealing",
        action="store_true",
        help="linearly anneal learning rate from the configured value to 0",
    )
    parser.add_argument("--ent_coef", type=float, default=0.001, help="entropy coefficient")
    parser.add_argument("--val_coef", type=float, default=0.6, help="val coefficient")
    parser.add_argument("--epoch_num", type=int, default=4, help="epoch number")
    parser.add_argument("--ppo_eps", type=float, default=0.2, help="PPO policy clip range")
    parser.add_argument(
        "--value_clip",
        type=float,
        default=2.0,
        help="value-function clip range for PPO-like algorithms",
    )
    parser.add_argument("--gae_normalize", action="store_true")
    parser.add_argument(
        "--gae_normalize_scope",
        type=str,
        default="batch",
        choices=["batch", "minibatch"],
        help="advantage normalization scope when --gae_normalize is set "
        "(batch: whole rollout, once; minibatch: per-minibatch, PPO2-style)",
    )
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
    policy_kwargs = {
        "node": args.node,
        "hidden_n": args.hidden_n,
        "embedding_mode": "normal",
    }
    return env_builder, policy_kwargs


def _common(a):
    return {
        "num_workers": a.worker,
        "gamma": a.gamma,
        "learning_rate": a.learning_rate,
        "batch_size": a.batch,
        "val_coef": a.val_coef,
        "ent_coef": a.ent_coef,
        "use_entropy_adv_shaping": a.use_entropy_adv_shaping,
        "log_dir": a.logdir,
        "optimizer": a.optimizer,
        "optimizer_eps": a.optimizer_eps,
        "max_grad_norm": a.max_grad_norm,
        "lr_annealing": a.lr_annealing,
        "seed": a.seed,
    }


def _ppo_like(a):
    return {
        **_common(a),
        "lamda": a.lamda,
        "gae_normalize": a.gae_normalize,
        "gae_normalize_scope": a.gae_normalize_scope,
        "minibatch_size": a.mini_batch,
        "epoch_num": a.epoch_num,
    }


def _ppo_build(a):
    return {
        **_ppo_like(a),
        "ppo_eps": a.ppo_eps,
        "value_clip": a.value_clip,
    }


def _tppo_build(a):
    return {
        **_ppo_like(a),
        "value_clip": a.value_clip,
    }


ALGOS = {
    "A2C": AlgoSpec(A2C, "ac", _common),
    "PPO": AlgoSpec(PPO, "ac", _ppo_build),
    "TPPO": AlgoSpec(TPPO, "ac", _tppo_build),
    "SPO": AlgoSpec(SPO, "ac", _ppo_build),
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
