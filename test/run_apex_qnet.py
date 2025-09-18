import argparse
import multiprocessing as mp
import os

import ray

from jax_baselines.APE_X.worker import Ape_X_Worker
from jax_baselines.C51.apex_c51 import APE_X_C51
from jax_baselines.common.env_builder import get_env_builder
from jax_baselines.DQN.apex_dqn import APE_X_DQN
from jax_baselines.IQN.apex_iqn import APE_X_IQN
from jax_baselines.QRDQN.apex_qrdqn import APE_X_QRDQN

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="Cartpole-v1", help="environment")
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
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/apex/", help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max", type=float, default=250, help="c51 max")
    parser.add_argument("--min", type=float, default=-250, help="c51 min")
    parser.add_argument("--n_support", type=int, default=200, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--delta", type=float, default=0.001, help="delta for QRDQN,IQN,FQF")
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
    embedding_mode = "normal"

    manger = mp.get_context().Manager()

    ray.init(num_cpus=args.worker + 2, num_gpus=0)

    env_builder, env_info = get_env_builder(env_name)
    workers = [Ape_X_Worker.remote(env_builder) for i in range(args.worker)]

    env_type = env_info["env_type"]
    env_name = env_info["env_id"]

    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "embedding_mode": embedding_mode}

    if args.algo == "DQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.dqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.dqn_builder import model_builder_maker

        agent = APE_X_DQN(
            workers,
            model_builder_maker,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_num=args.batch_num,
            mini_batch_size=args.batch_size,
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
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
        )
    elif args.algo == "C51":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.c51_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.c51_builder import model_builder_maker

        agent = APE_X_C51(
            workers,
            model_builder_maker,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_num=args.batch_num,
            mini_batch_size=args.batch_size,
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
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            categorial_max=args.max,
            categorial_min=args.min,
        )
    elif args.algo == "QRDQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.qrdqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.qrdqn_builder import model_builder_maker

        agent = APE_X_QRDQN(
            workers,
            model_builder_maker,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_num=args.batch_num,
            mini_batch_size=args.batch_size,
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
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            n_support=args.n_support,
            delta=args.delta,
        )
    elif args.algo == "IQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.iqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.iqn_builder import model_builder_maker

        agent = APE_X_IQN(
            workers,
            model_builder_maker,
            manger,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_num=args.batch_num,
            mini_batch_size=args.batch_size,
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
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            n_support=args.n_support,
            delta=args.delta,
            CVaR=args.CVaR,
        )

    agent.learn(int(args.steps))
