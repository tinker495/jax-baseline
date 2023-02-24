from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from tqdm import trange
import numpy as np


env_name = "../../env/Walker.x86_64"

engine_configuration_channel = EngineConfigurationChannel()
channel = EnvironmentParametersChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=20,capture_frame_rate=30)
env = UnityEnvironment(file_name=env_name,worker_id=0,no_graphics=True,side_channels=[engine_configuration_channel,channel])


env.reset()
group_name = list(env.behavior_specs.keys())[0]
group_spec = env.behavior_specs[group_name]
action_size = [group_spec.action_spec.continuous_size]

env.step()
dec, term = env.get_steps(group_name)
worker_size = len(dec.agent_id)
actions = np.random.uniform(-1.0,1.0,size=(worker_size,action_size[0]))

for i in trange(int(1e7)):
    env.set_actions(group_name, ActionTuple(continuous=actions))
    env.step()
    dec, term = env.get_steps(group_name)