import os
import argparse
import gymnasium as gym

from haiku_baselines.DDPG.ddpg import DDPG
from haiku_baselines.TD3.td3 import TD3
from haiku_baselines.TD4_QR.td4_qr import TD4_QR
from haiku_baselines.TD4_IQN.td4_iqn import TD4_IQN
from haiku_baselines.SAC.sac import SAC
from haiku_baselines.TQC.tqc import TQC
from haiku_baselines.IQA_TQC.iqa_tqc import IQA_TQC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0000625, help='learning rate')
    parser.add_argument('--env', type=str, default="Pendulum-v0", help='environment')
    parser.add_argument('--worker_id', type=int, default=0, help="unlty ml agent's worker id")
    parser.add_argument('--worker', type=int,default=1, help='gym_worker_size')
    parser.add_argument('--algo', type=str, default="DDPG", help='algo ID')
    parser.add_argument('--gamma', type=float, default=0.995, help='gamma')
    parser.add_argument('--target_update_tau', type=float, default=2e-3, help='target update intervals')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--buffer_size', type=float, default=2000000, help='buffer_size')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--n_step', type=int, default=1, help='n step setting when n > 1 is n step td method')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_support', type=int,default=25, help='n_support for QRDQN,IQN,FQF')
    parser.add_argument('--mixture', type=str,default='truncated', help='mixture type')
    parser.add_argument('--quantile_drop', type=float,default=0.1, help='quantile_drop ratio')
    parser.add_argument('--node', type=int,default=256, help='network node number')
    parser.add_argument('--hidden_n', type=int,default=2, help='hidden layer number')
    parser.add_argument('--action_noise', type=float,default=0.1, help='action_noise')
    parser.add_argument('--optimizer', type=str,default='adam', help='optimaizer')
    parser.add_argument('--train_freq', type=int, default=1, help='train_frequancy')
    parser.add_argument('--critic_num', type=int,default=2, help='tqc critic number')
    parser.add_argument('--ent_coef', type=str,default='auto', help='sac entropy coefficient')
    parser.add_argument('--learning_starts', type=int, default=5000, help='learning start')
    parser.add_argument('--risk_avoidance', type=float,default=0.0, help='risk_avoidance')
    parser.add_argument('--time_scale', type=float,default=12.0, help='risk_avoidance')
    args = parser.parse_args() 
    env_name = args.env
    cnn_mode = "normal"
    if os.path.exists(env_name):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
        
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        engine_configuration_channel.set_configuration_parameters(time_scale=args.time_scale,capture_frame_rate=30)
        env = UnityEnvironment(file_name=env_name,worker_id=args.worker_id,no_graphics=False,side_channels=[engine_configuration_channel,channel])
        env_name = env_name.split('/')[-1].split('.')[0]
        env_type = "unity"
    else:
        #import mujoco_py
        if args.worker > 1:
            from haiku_baselines.common.worker import gymnasium as gymMultiworker
            env = gymMultiworker(env_name, worker_num = args.worker)
        else:
            env = gym.make(env_name)
        env_type = "gym"

    
    policy_kwargs = {'node': args.node,
                     'hidden_n': args.hidden_n,
                     'cnn_mode': cnn_mode}
        
    if args.algo == "DDPG":
        agent = DDPG(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts,
                    prioritized_replay = args.per, exploration_final_eps = args.eps, n_step = args.n_step, train_freq=args.train_freq, seed = args.seed,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "TD3":
        agent = TD3(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts,
                    prioritized_replay = args.per, action_noise = args.action_noise, n_step = args.n_step, train_freq=args.train_freq, seed = args.seed,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "TD4_QR":
        agent = TD4_QR(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts, risk_avoidance = args.risk_avoidance, quantile_drop=args.quantile_drop,
                    prioritized_replay = args.per, action_noise = args.action_noise, n_step = args.n_step, train_freq=args.train_freq, seed = args.seed,
                    n_support = args.n_support, mixture_type = args.mixture,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "TD4_IQN":
        agent = TD4_IQN(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts, risk_avoidance = args.risk_avoidance,
                    prioritized_replay = args.per, action_noise = args.action_noise, n_step = args.n_step, train_freq=args.train_freq, seed = args.seed,
                    n_support = args.n_support, mixture_type = args.mixture,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "SAC":
        agent = SAC(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts,
                    prioritized_replay = args.per, n_step = args.n_step, train_freq=args.train_freq, ent_coef = args.ent_coef, seed = args.seed,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "TQC":
        agent = TQC(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts, risk_avoidance = args.risk_avoidance, quantile_drop=args.quantile_drop,
                    prioritized_replay = args.per, n_step = args.n_step, train_freq=args.train_freq, ent_coef = args.ent_coef, seed = args.seed,
                    n_support = args.n_support, critic_num = args.critic_num, mixture_type = args.mixture,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "IQA":
        agent = IQA_TQC(env, gamma=args.gamma, learning_rate=args.learning_rate, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_tau = args.target_update_tau, learning_starts=args.learning_starts, risk_avoidance = args.risk_avoidance, quantile_drop=args.quantile_drop,
                    prioritized_replay = args.per, n_step = args.n_step, train_freq=args.train_freq, ent_coef = args.ent_coef, seed = args.seed,
                    n_support = args.n_support, critic_num = args.critic_num, mixture_type = args.mixture,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
        



    agent.learn(int(args.steps))
    
    agent.test()
    env.close()