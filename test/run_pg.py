import os
import argparse
import gym

from haiku_baselines.A2C.a2c import A2C
from haiku_baselines.PPO.ppo import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Pendulum-v1", help='environment')
    parser.add_argument('--worker_id', type=int, default=0, help="unlty ml agent's worker id")
    parser.add_argument('--worker', type=int,default=1, help='gym_worker_size')
    parser.add_argument('--algo', type=str, default="A2C", help='algo ID')
    parser.add_argument('--gamma', type=float, default=0.995, help='gamma')
    parser.add_argument('--lamda', type=float, default=0.95, help='gae lamda')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--mini_batch', type=int, default=32, help='batch size')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--node', type=int,default=256, help='network node number')
    parser.add_argument('--hidden_n', type=int,default=2, help='hidden layer number')
    parser.add_argument('--optimizer', type=str,default='rmsprop', help='optimaizer')
    parser.add_argument('--ent_coef', type=float,default=0.001, help='entropy coefficient')
    parser.add_argument('--val_coef', type=float,default=0.6, help='val coefficient')
    parser.add_argument('--gae_normalize', dest='gae_normalize', action='store_true')
    parser.add_argument('--no_gae_normalize', dest='gae_normalize', action='store_false')
    parser.set_defaults(gae_normalize=False)

    args = parser.parse_args() 
    env_name = args.env
    cnn_mode = "normal"
    if os.path.exists(env_name):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
        
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        engine_configuration_channel.set_configuration_parameters(time_scale=12.0,capture_frame_rate=30)
        env = UnityEnvironment(file_name=env_name,worker_id=args.worker_id,no_graphics=False,side_channels=[engine_configuration_channel,channel])
        env_name = env_name.split('/')[-1].split('.')[0]
        env_type = "unity"
    else:
        import mujoco_py
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
        
    if args.algo == "A2C":
        agent = A2C(env, gamma=args.gamma, batch_size = args.batch, val_coef = args.val_coef, ent_coef = args.ent_coef,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)
    if args.algo == "PPO":
        agent = PPO(env, gamma=args.gamma, lamda=args.lamda, gae_normalize=args.gae_normalize, batch_size = args.batch ,minibatch_size = args.mini_batch, 
                    val_coef = args.val_coef, ent_coef = args.ent_coef,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs, optimizer=args.optimizer)



    agent.learn(int(args.steps))
    
    agent.test()