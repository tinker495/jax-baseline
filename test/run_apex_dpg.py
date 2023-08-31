import os
import argparse
import gymnasium as gym
import ray
import multiprocessing as mp

from jax_baselines.APE_X.dpg_worker import Ape_X_Worker
from jax_baselines.DDPG.apex_ddpg import APE_X_DDPG
from jax_baselines.TD3.apex_td3 import APE_X_TD3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--env", type=str, default="Pendulum-v0", help="environment")
    parser.add_argument("--worker_id", type=int, default=0, help="unlty ml agent's worker id")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="DDPG", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument(
        "--target_update_tau", type=float, default=2e-3, help="target update intervals"
    )
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=100000, help="buffer_size")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/", help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_support", type=int, default=25, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--mixture", type=str, default="truncated", help="mixture type")
    parser.add_argument("--quantile_drop", type=float, default=0.1, help="quantile_drop ratio")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--action_noise", type=float, default=0.1, help="action_noise")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient_steps")
    parser.add_argument("--critic_num", type=int, default=2, help="tqc critic number")
    parser.add_argument("--ent_coef", type=str, default="auto", help="sac entropy coefficient")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--initial_eps", type=float, default=0.4, help="initial epsilon")
    parser.add_argument("--eps_decay", type=float, default=3, help="exploration fraction")
    parser.add_argument("--cvar", type=float, default=1.0, help="cvar")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    args = parser.parse_args()
    env_name = args.env
    cnn_mode = "normal"

    manger = mp.get_context().Manager()

    ray.init(num_cpus=args.worker + 2, num_gpus=0)

    workers = [Ape_X_Worker.remote(env_name) for i in range(args.worker)]

    env_type = "gym"

    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "cnn_mode": cnn_mode}

    if args.algo == "DDPG":
        agent = APE_X_DDPG(
            workers,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            exploration_initial_eps=args.initial_eps,
            exploration_decay=args.eps_decay,
            n_step=args.n_step,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    elif args.algo == "TD3":
        agent = APE_X_TD3(
            workers,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            exploration_initial_eps=args.initial_eps,
            exploration_decay=args.eps_decay,
            n_step=args.n_step,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )

    agent.learn(int(args.steps))
