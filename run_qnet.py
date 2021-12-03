import os
import argparse
import gym

from haiku_baselines.DQN.dqn import DQN
from haiku_baselines.C51.c51 import C51
from haiku_baselines.QRDQN.qrdqn import QRDQN
from haiku_baselines.IQN.iqn import IQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Cartpole-v1", help='environment')
    parser.add_argument('--algo', type=str, default="DQN", help='algo ID')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--target_update', type=int, default=2000, help='target update intervals')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--buffer_size', type=float, default=100000, help='buffer_size')
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--dueling',action='store_true')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--noisynet', action='store_true')
    parser.add_argument('--n_step', type=int, default=1, help='n step setting when n > 1 is n step td method')
    parser.add_argument('--munchausen', action='store_true')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max', type=float, default=250, help='c51 max')
    parser.add_argument('--min', type=float, default=-250, help='c51 min')
    parser.add_argument('--n_support', type=int,default=32, help='n_support for QRDQN,IQN,FQF')
    parser.add_argument('--CVaR', type=float, default=1.0, help='IQN risk avoiding factor')
    parser.add_argument('--node', type=int,default=256, help='network node number')
    parser.add_argument('--hidden_n', type=int,default=2, help='hidden layer number')
    parser.add_argument('--final_eps', type=float,default=0.1, help='final epsilon')
    parser.add_argument('--worker', type=int,default=1, help='gym_worker_size')
    args = parser.parse_args() 
    env_name = args.env
    cnn_mode = "normal"
    if os.path.exists(env_name):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
        
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        engine_configuration_channel.set_configuration_parameters(time_scale=12.0,capture_frame_rate=50)
        env = UnityEnvironment(file_name=env_name,no_graphics=False, side_channels=[engine_configuration_channel,channel],timeout_wait=10000)
        env_name = env_name.split('/')[-1].split('.')[0]
        env_type = "unity"
    else:
        if args.worker > 1:
            from haiku_baselines.common.worker import gymMultiworker
            env = gymMultiworker(env_name, worker_num = args.worker)
        else:
            from haiku_baselines.common.atari_wrappers import make_wrap_atari,get_env_type
            env_type, env_id = get_env_type(env_name)
            if env_type == 'atari':
                env = make_wrap_atari(env_name)
            else:
                env = gym.make(env_name)
        env_type = "gym"

    
    policy_kwargs = {'node': args.node,
                     'hidden_n': args.hidden_n,
                     'cnn_mode': cnn_mode}
        
    if args.algo == "DQN":
        agent = DQN(env, gamma=args.gamma, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_freq = args.target_update,
                    prioritized_replay = args.per, double_q = args.double, dueling_model = args.dueling, exploration_final_eps = args.final_eps,
                    param_noise = args.noisynet, n_step = args.n_step, munchausen = args.munchausen,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "C51":
        agent = C51(env, gamma=args.gamma, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_freq = args.target_update,
                    prioritized_replay = args.per, double_q = args.double, dueling_model = args.dueling, exploration_final_eps = args.final_eps,
                    param_noise = args.noisynet, n_step = args.n_step, munchausen = args.munchausen,
                    categorial_max = args.max, categorial_min = args.min,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "QRDQN":
        agent = QRDQN(env, gamma=args.gamma, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_freq = args.target_update,
                    prioritized_replay = args.per, double_q = args.double, dueling_model = args.dueling, exploration_final_eps = args.final_eps,
                    param_noise = args.noisynet, n_step = args.n_step, munchausen = args.munchausen,
                    n_support = args.n_support,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "IQN":
        agent = IQN(env, gamma=args.gamma, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_freq = args.target_update,
                    prioritized_replay = args.per, double_q = args.double, dueling_model = args.dueling, exploration_final_eps = args.final_eps,
                    param_noise = args.noisynet, n_step = args.n_step, munchausen = args.munchausen,
                    n_support = args.n_support,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)


    agent.learn(int(args.steps))