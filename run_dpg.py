import os
import argparse
import gym

from haiku_baselines.DDPG.ddpg import DDPG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Cartpole-v1", help='environment')
    parser.add_argument('--algo', type=str, default="DQN", help='algo ID')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--target_update_tau', type=float, default=2e-4, help='target update intervals')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--buffer_size', type=float, default=100000, help='buffer_size')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--n_step', type=int, default=1, help='n step setting when n > 1 is n step td method')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max', type=float, default=250, help='c51 max')
    parser.add_argument('--min', type=float, default=-250, help='c51 min')
    parser.add_argument('--n_support', type=int,default=32, help='n_support for QRDQN,IQN,FQF')
    parser.add_argument('--node', type=int,default=256, help='network node number')
    parser.add_argument('--hidden_n', type=int,default=2, help='hidden layer number')
    parser.add_argument('--final_eps', type=float,default=0.1, help='final epsilon')
    parser.add_argument('--worker', type=int,default=1, help='gym_worker_size')
    parser.add_argument('--optimizer', type=str,default='adamw', help='optimaizer')
    parser.add_argument('--train_freq', type=int, default=1, help='train_frequancy')
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
            env = gym.make(env_name)
        env_type = "gym"

    
    policy_kwargs = {'node': args.node,
                     'hidden_n': args.hidden_n,
                     'cnn_mode': cnn_mode}
        
    if args.algo == "DDPG":
        agent = DDPG(env, gamma=args.gamma, batch_size = args.batch, buffer_size= int(args.buffer_size), target_network_update_freq = args.target_update_tau,
                    prioritized_replay = args.per, exploration_final_eps = args.final_eps, n_step = args.n_step, train_freq=args.train_freq,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)


    agent.learn(int(args.steps))
    
    agent.test()