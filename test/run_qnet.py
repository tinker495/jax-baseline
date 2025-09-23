import argparse
import os

from jax_baselines.BBF.bbf import BBF
from jax_baselines.BBF.hl_gauss_bbf import HL_GAUSS_BBF
from jax_baselines.C51.c51 import C51
from jax_baselines.C51.hl_gauss_c51 import HL_GAUSS_C51
from jax_baselines.common.env_builder import get_env_builder
from jax_baselines.DQN.dqn import DQN
from jax_baselines.FQF.fqf import FQF
from jax_baselines.IQN.iqn import IQN
from jax_baselines.QRDQN.qrdqn import QRDQN
from jax_baselines.SPR.hl_gauss_spr import HL_GAUSS_SPR
from jax_baselines.SPR.spr import SPR

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="Q_network", help="experiment name")
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="Cartpole-v1", help="environment")
    parser.add_argument("--algo", type=str, default="DQN", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--target_update", type=int, default=2000, help="target update intervals")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=200000, help="buffer_size")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--noisynet", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--off_policy_fix", action="store_true")
    parser.add_argument("--munchausen", action="store_true")
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/qnet/", help="log file dir")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max", type=float, default=10, help="c51 max")
    parser.add_argument("--min", type=float, default=-10, help="c51 min")
    parser.add_argument("--n_support", type=int, default=32, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="huber loss delta  for QRDQN,IQN,FQF"
    )
    parser.add_argument("--CVaR", type=float, default=1.0, help="IQN risk avoiding factor")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--final_eps", type=float, default=0.1, help="final epsilon")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimaizer")
    parser.add_argument("--train_freq", type=int, default=1, help="train_frequancy")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient steps")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument(
        "--exploration_fraction", type=float, default=0.3, help="exploration fraction"
    )
    parser.add_argument("--clip_rewards", action="store_true")
    parser.add_argument("--compress_memory", action="store_true")
    parser.add_argument("--hl_gauss", action="store_true")
    parser.add_argument("--scaled_by_reset", action="store_true")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    parser.add_argument("--use_checkpointing", action="store_true")
    args = parser.parse_args()
    env_name = args.env
    embedding_mode = "normal"
    env_builder, env_info = get_env_builder(
        env_name, timescale=args.time_scale, capture_frame_rate=args.capture_frame_rate
    )
    env_name = env_info["env_id"]
    env_type = env_info["env_type"]
    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n}

    if args.model_lib == "equinox" and args.algo not in {"DQN", "C51", "QRDQN"}:
        raise ValueError(
            "Equinox backend currently supports only DQN, C51, and QRDQN in run_qnet."
        )

    if args.algo == "DQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.dqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.dqn_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.dqn_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")
        agent = DQN(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            prioritized_replay=args.per,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_final_eps=args.final_eps,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            exploration_fraction=args.exploration_fraction,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            use_checkpointing=args.use_checkpointing,
        )
    elif args.algo == "C51":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.c51_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.c51_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.c51_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")

        if args.hl_gauss:
            agent = HL_GAUSS_C51(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                target_network_update_freq=args.target_update,
                prioritized_replay=args.per,
                double_q=args.double,
                dueling_model=args.dueling,
                exploration_final_eps=args.final_eps,
                param_noise=args.noisynet,
                n_step=args.n_step,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                train_freq=args.train_freq,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                exploration_fraction=args.exploration_fraction,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )
        else:
            agent = C51(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                target_network_update_freq=args.target_update,
                prioritized_replay=args.per,
                double_q=args.double,
                dueling_model=args.dueling,
                exploration_final_eps=args.final_eps,
                param_noise=args.noisynet,
                n_step=args.n_step,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                train_freq=args.train_freq,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                exploration_fraction=args.exploration_fraction,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )
    elif args.algo == "QRDQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.qrdqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.qrdqn_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.qrdqn_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")
        agent = QRDQN(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            prioritized_replay=args.per,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_final_eps=args.final_eps,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            delta=args.delta,
            n_support=args.n_support,
            exploration_fraction=args.exploration_fraction,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            use_checkpointing=args.use_checkpointing,
        )
    elif args.algo == "IQN":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.iqn_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.iqn_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.iqn_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")
        agent = IQN(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            prioritized_replay=args.per,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_final_eps=args.final_eps,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            delta=args.delta,
            n_support=args.n_support,
            exploration_fraction=args.exploration_fraction,
            CVaR=args.CVaR,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            use_checkpointing=args.use_checkpointing,
        )
    elif args.algo == "FQF":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.fqf_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.fqf_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.fqf_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")
        agent = FQF(
            env_builder,
            model_builder_maker=model_builder_maker,
            num_workers=args.worker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=args.target_update,
            prioritized_replay=args.per,
            double_q=args.double,
            dueling_model=args.dueling,
            exploration_final_eps=args.final_eps,
            param_noise=args.noisynet,
            n_step=args.n_step,
            munchausen=args.munchausen,
            gradient_steps=args.gradient_steps,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            delta=args.delta,
            n_support=args.n_support,
            exploration_fraction=args.exploration_fraction,
            log_dir=args.logdir,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
            compress_memory=args.compress_memory,
            use_checkpointing=args.use_checkpointing,
        )
    elif args.algo == "SPR":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.spr_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.qnet.spr_builder import model_builder_maker
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.spr_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")

        if args.hl_gauss:
            agent = HL_GAUSS_SPR(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                off_policy_fix=args.off_policy_fix,
                scaled_by_reset=args.scaled_by_reset,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )
        else:
            agent = SPR(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                off_policy_fix=args.off_policy_fix,
                scaled_by_reset=args.scaled_by_reset,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )

    elif args.algo == "BBF":
        if args.model_lib == "flax":
            from model_builder.flax.qnet.bbf_builder import model_builder_maker
        elif args.model_lib == "haiku":
            raise NotImplementedError
        elif args.model_lib == "equinox":
            from model_builder.equinox.qnet.bbf_builder import model_builder_maker
        else:
            raise ValueError(f"Unsupported model_lib: {args.model_lib}")

        if args.hl_gauss:
            agent = HL_GAUSS_BBF(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                exploration_final_eps=args.final_eps,
                param_noise=args.noisynet,
                off_policy_fix=args.off_policy_fix,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                train_freq=args.train_freq,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                exploration_fraction=args.exploration_fraction,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )
        else:
            agent = BBF(
                env_builder,
                model_builder_maker=model_builder_maker,
                num_workers=args.worker,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                batch_size=args.batch,
                buffer_size=int(args.buffer_size),
                exploration_final_eps=args.final_eps,
                param_noise=args.noisynet,
                off_policy_fix=args.off_policy_fix,
                munchausen=args.munchausen,
                gradient_steps=args.gradient_steps,
                train_freq=args.train_freq,
                learning_starts=args.learning_starts,
                categorial_max=args.max,
                categorial_min=args.min,
                exploration_fraction=args.exploration_fraction,
                log_dir=args.logdir,
                policy_kwargs=policy_kwargs,
                optimizer=args.optimizer,
                compress_memory=args.compress_memory,
                use_checkpointing=args.use_checkpointing,
            )

    agent.learn(int(args.steps), experiment_name=args.experiment_name)

    agent.test()
