import argparse

from jax_baselines.A2C.a2c import A2C
from jax_baselines.common.env_builder import get_env_builder
from jax_baselines.PPO.ppo import PPO
from jax_baselines.SPO.spo import SPO
from jax_baselines.TPPO.tppo import TPPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="PG", help="experiment name")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="environment")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--worker_id", type=int, default=0, help="unlty ml agent's worker id")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="A2C", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument("--lamda", type=float, default=0.95, help="gae lamda")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--mini_batch", type=int, default=32, help="batch size")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/pg/", help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="entropy coefficient")
    parser.add_argument("--val_coef", type=float, default=0.6, help="val coefficient")
    parser.add_argument("--gae_normalize", dest="gae_normalize", action="store_true")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    parser.set_defaults(gae_normalize=False)

    args = parser.parse_args()
    env_name = args.env
    embedding_mode = "normal"
    env_builder, env_info = get_env_builder(
        env_name, timescale=args.time_scale, capture_frame_rate=args.capture_frame_rate
    )
    env_name = env_info["env_id"]
    env_type = env_info["env_type"]
    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n}

    if args.model_lib == "flax":
        from model_builder.flax.ac.ac_builder import model_builder_maker
    elif args.model_lib == "haiku":
        from model_builder.haiku.ac.ac_builder import model_builder_maker

    if args.algo == "A2C":
        agent = A2C(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            batch_size=args.batch,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "PPO":
        agent = PPO(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            lamda=args.lamda,
            gae_normalize=args.gae_normalize,
            batch_size=args.batch,
            minibatch_size=args.mini_batch,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TPPO":
        agent = TPPO(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            lamda=args.lamda,
            gae_normalize=args.gae_normalize,
            batch_size=args.batch,
            minibatch_size=args.mini_batch,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "SPO":
        agent = SPO(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            lamda=args.lamda,
            gae_normalize=args.gae_normalize,
            batch_size=args.batch,
            minibatch_size=args.mini_batch,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )

    agent.learn(int(args.steps), experiment_name=args.experiment_name)

    agent.test()
