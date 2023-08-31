import os
import argparse
import gymnasium as gym
import ray
import multiprocessing as mp

from jax_baselines.IMPALA.worker import Impala_Worker
from jax_baselines.A2C.impala import IMPALA
from jax_baselines.PPO.impala_ppo import IMPALA_PPO
from jax_baselines.TPPO.impala_tppo import IMPALA_TPPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="environment")
    parser.add_argument("--worker_id", type=int, default=0, help="unlty ml agent's worker id")
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
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/", help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--optimizer", type=str, default="rmsprop", help="optimaizer")
    parser.add_argument("--ent_coef", type=float, default=0.1, help="entropy coefficient")
    parser.add_argument("--val_coef", type=float, default=0.6, help="val coefficient")
    parser.add_argument("--gae_normalize", dest="gae_normalize", action="store_true")
    parser.add_argument("--no_gae_normalize", dest="gae_normalize", action="store_false")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    parser.add_argument("--mu_ratio", type=float, default=0.0, help="impala ppo mu ratio")
    args = parser.parse_args()
    env_name = args.env
    cnn_mode = "normal"
    # cnn_mode = "minimum"
    # cnn_mode = "none"

    manger = mp.get_context().Manager()

    ray.init(num_cpus=args.worker + 4, num_gpus=0)

    workers = [Impala_Worker.remote(env_name) for i in range(args.worker)]

    env_type = "gym"

    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "cnn_mode": cnn_mode}

    if args.algo == "A2C":
        agent = IMPALA(
            workers,
            manger,
            gamma=args.gamma,
            lamda=args.lamda,
            learning_rate=args.learning_rate,
            update_freq=args.update_freq,
            batch_size=args.batch,
            sample_size=args.sample_size,
            buffer_size=int(args.buffer_size),
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            rho_max=args.rho_max,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
        )

    elif args.algo == "PPO":
        agent = IMPALA_PPO(
            workers,
            manger,
            gamma=args.gamma,
            lamda=args.lamda,
            learning_rate=args.learning_rate,
            update_freq=args.update_freq,
            batch_size=args.batch,
            sample_size=args.sample_size,
            buffer_size=int(args.buffer_size),
            mu_ratio=args.mu_ratio,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            rho_max=args.rho_max,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
        )

    elif args.algo == "TPPO":
        agent = IMPALA_TPPO(
            workers,
            manger,
            gamma=args.gamma,
            lamda=args.lamda,
            learning_rate=args.learning_rate,
            update_freq=args.update_freq,
            batch_size=args.batch,
            sample_size=args.sample_size,
            buffer_size=int(args.buffer_size),
            mu_ratio=args.mu_ratio,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            val_coef=args.val_coef,
            ent_coef=args.ent_coef,
            rho_max=args.rho_max,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
        )

    agent.learn(int(args.steps))
