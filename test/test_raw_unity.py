from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from tqdm import trange
import numpy as np

engine_configuration_channel = EngineConfigurationChannel()
channel = EnvironmentParametersChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=20.0, capture_frame_rate=1)
env = UnityEnvironment(
    file_name="../env/Walker.x86_64",
    worker_id=0,
    no_graphics=False,
    side_channels=[engine_configuration_channel, channel],
)
env_name = "3DBall"
env_type = "unity"

print("unity-ml agent environmet")
env.reset()
group_name = list(env.behavior_specs.keys())[0]
group_spec = env.behavior_specs[group_name]
env.step()
dec, term = env.get_steps(group_name)
group_name = group_name

observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
if group_spec.action_spec.continuous_size == 0:
    action_size = [branch for branch in group_spec.action_spec.discrete_branches]
    action_type = "discrete"
    conv_action = lambda a: ActionTuple(discrete=a)
else:
    action_size = [group_spec.action_spec.continuous_size]
    action_type = "continuous"
    conv_action = lambda a: ActionTuple(
        continuous=np.clip(a, -3.0, 3.0) / 3.0
    )  # np.clip(a, -3.0, 3.0) / 3.0)
worker_size = len(dec.agent_id)
env_type = "unity"

env.reset()
env.step()
dec, term = env.get_steps(group_name)
ldec = len(dec)
for step in trange(100000):
    action = np.random.uniform(-1, 1, size=(ldec, action_size[0]))
    env.set_actions(group_name, conv_action(action))
    env.step()
    dec, term = env.get_steps(group_name)
    while len(dec) == 0:
        env.step()
        dec, term = env.get_steps(group_name)
    ldec = len(dec)

env.close()
