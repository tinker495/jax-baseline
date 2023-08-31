import os
import argparse
import gymnasium as gym
import ray
import multiprocessing as mp

from haiku_baselines.APE_X.worker import Ape_X_Worker
from haiku_baselines.DQN.apex_dqn import APE_X_DQN
from haiku_baselines.C51.apex_c51 import APE_X_C51
from haiku_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--env", type=str, default="Cartpole-v1", help="environment")
    parser.add_argument("--algo", type=str, default="DQN", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--target_update", type=int, default=2000, help="target update intervals")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
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
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/", help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max", type=float, default=250, help="c51 max")
    parser.add_argument("--min", type=float, default=-250, help="c51 min")
    parser.add_argument("--n_support", type=int, default=200, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--delta", type=float, default=0.5, help="network node number")
    parser.add_argument("--CVaR", type=float, default=1.0, help="IQN risk avoiding factor")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--final_eps", type=float, default=0.1, help="final epsilon")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient steps")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--initial_eps", type=float, default=0.4, help="initial epsilon")
    parser.add_argument("--eps_decay", type=float, default=3, help="exploration fraction")
    parser.add_argument("--clip_rewards", action="store_true")
    parser.add_argument("--compress_memory", action="store_true")
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

    if args.algo == "DQN":
        agent = APE_X_DQN(
            workers,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_initial_eps=args.initial_eps,
            exploration_decay=args.eps_decay,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
        )
    elif args.algo == "C51":
        agent = APE_X_C51(
            workers,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_initial_eps=args.initial_eps,
            exploration_decay=args.eps_decay,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            categorial_max=args.max,
            categorial_min=args.min,
        )
    elif args.algo == "QRDQN":
        agent = APE_X_QRDQN(
            workers,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_initial_eps=args.initial_eps,
            exploration_decay=args.eps_decay,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            n_support=args.n_support,
            delta=args.delta,
        )

    agent.learn(int(args.steps))
