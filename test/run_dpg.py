import argparse

import gymnasium as gym

from jax_baselines.common.env_builer import get_env_builder
from jax_baselines.CrossQ.crossq import CrossQ
from jax_baselines.DAC.dac import DAC
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.SAC.sac import SAC
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TQC.tqc import TQC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="DPG", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
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
    parser.add_argument("--per", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--scaled_by_reset", action="store_true")
    parser.add_argument("--simba", action="store_true")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/dpg/", help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_support", type=int, default=25, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--mixture", type=str, default="truncated", help="mixture type")
    parser.add_argument("--quantile_drop", type=float, default=0.1, help="quantile_drop ratio")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--action_noise", type=float, default=0.1, help="action_noise")
    parser.add_argument("--optimizer", type=str, default="adopt", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient_steps")
    parser.add_argument("--train_freq", type=int, default=1, help="train_frequancy")
    parser.add_argument("--critic_num", type=int, default=2, help="tqc critic number")
    parser.add_argument("--ent_coef", type=str, default="auto", help="sac entropy coefficient")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    args = parser.parse_args()
    env_name = args.env
    embedding_mode = "normal"
    env_builder, env_info = get_env_builder(
        env_name, timescale=args.time_scale, capture_frame_rate=args.capture_frame_rate
    )
    env_name = env_info["env_id"]
    env_type = env_info["env_type"]

    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "embedding_mode": embedding_mode}

    if args.algo == "DDPG":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_ddpg_builder import (
                    model_builder_maker,
                )
            else:
                from model_builder.flax.dpg.ddpg_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.ddpg_builder import model_builder_maker
        agent = DDPG(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            n_step=args.n_step,
            train_freq=args.train_freq,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TD3":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_td3_builder import model_builder_maker
            else:
                from model_builder.flax.dpg.td3_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.td3_builder import model_builder_maker
        agent = TD3(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            action_noise=args.action_noise,
            n_step=args.n_step,
            train_freq=args.train_freq,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "SAC":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_sac_builder import model_builder_maker
            else:
                from model_builder.flax.dpg.sac_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.sac_builder import model_builder_maker
        agent = SAC(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "CrossQ":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_crossq_builder import (
                    model_builder_maker,
                )
            else:
                from model_builder.flax.dpg.crossq_builder import model_builder_maker
        elif args.model_lib == "haiku":
            pass
            # from model_builder.haiku.dpg.crossq_builder import model_builder_maker
        agent = CrossQ(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "DAC":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_dac_builder import model_builder_maker
            else:
                from model_builder.flax.dpg.dac_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.dac_builder import model_builder_maker
        agent = DAC(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TQC":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_tqc_builder import model_builder_maker
            else:
                from model_builder.flax.dpg.tqc_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.tqc_builder import model_builder_maker
        agent = TQC(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            quantile_drop=args.quantile_drop,
            prioritized_replay=args.per,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            n_support=args.n_support,
            critic_num=args.critic_num,
            mixture_type=args.mixture,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TD7":
        if args.model_lib == "flax":
            if args.simba:
                from model_builder.flax.dpg.simba_td7_builder import model_builder_maker
            else:
                from model_builder.flax.dpg.td7_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.td7_builder import model_builder_maker
        eval_env = gym.make(env_name)
        agent = TD7(
            env_builder,
            model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=250,
            learning_starts=args.learning_starts,
            action_noise=args.action_noise,
            train_freq=args.train_freq,
            scaled_by_reset=args.scaled_by_reset,
            simba=args.simba,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )

    agent.learn(int(args.steps), experiment_name=args.experiment_name)

    agent.test()
