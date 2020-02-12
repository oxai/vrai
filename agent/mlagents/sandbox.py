import matplotlib.pyplot as plt
import numpy as np
import sys

sys.version

from mlagents.envs.environment import UnityEnvironment
import mlagents
# mlagents.__spec__
mlagents.__path__
from mlagents import envs
import mlagents.envs.side_channel
from mlagents.envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
# ^ needs sudo pip3 install --upgrade . on ml-agents and ml-agents-envs of latest_release tag

'''Start the environment'''
'''UnityEnvironment launches and begins communication with the environment when instantiated. Environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python. '''
# env_name = "../envs/GridWorld"  # Name of the Unity environment binary to launch
env_name = None # to use the Unity editor
train_mode = True  # Whether to run the environment in training or inference mode
engine_configuration_channel = EngineConfigurationChannel()
# env = UnityEnvironment(base_port = 5006, file_name=env_name, side_channels = [engine_configuration_channel])
# mlagents.__spec__
env = UnityEnvironment(base_port = 5004, file_name=env_name, side_channels = [engine_configuration_channel])
# env = UnityEnvironment(base_port = 5004, file_name=env_name)
# env = UnityEnvironment(file_name=env_name, side_channels = [engine_configuration_channel])
# env = UnityEnvironment(base_port = 5004, file_name=env_name, side_channels = [engine_configuration_channel])
#%%

#Reset the environment
env.reset()

# Set the default brain to work with
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

# Set the time scale of the engine
engine_configuration_channel.set_configuration_parameters(time_scale = 3.0)

# env.reset()
# comm = env.get_communicator(2,5004,100)

'''Examine the observation and state spaces'''

'''
We can reset the environment to be provided with an initial set of observations and states for all the agents within the environment. In ML-Agents, states refer to a vector of variables corresponding to relevant aspects of the environment for an agent. Likewise, observations refer to a set of relevant pixel-wise visuals for an agent.
'''

# Get the state of the agents
step_result = env.get_step_result(group_name)

# Examine the number of observations per Agent
print("Number of observations : ", len(group_spec.observation_shapes))

# Examine the state space for the first observation for all agents
print("Agent state looks like: \n{}".format(step_result.obs[0]))

# Examine the state space for the first observation for the first agent
print("Agent state looks like: \n{}".format(step_result.obs[0][0]))

# Is there a visual observation ?
vis_obs = any([len(shape) == 3 for shape in group_spec.observation_shapes])
print("Is there a visual observation ?", vis_obs)

# Examine the visual observations
if vis_obs:
    vis_obs_index = next(i for i,v in enumerate(group_spec.observation_shapes) if len(v) == 3)
    print("Agent visual observation look like:")
    obs = step_result.obs[vis_obs_index]
    plt.imshow(obs[0,:,:,:])

'''Take random actions in the environment'''
'''
Once we restart an environment, we can step the environment forward and provide actions to all of the agents within the environment. Here we simply choose random actions based on the action_space_type of the default brain.

Once this cell is executed, 10 messages will be printed that detail how much reward will be accumulated for the next 10 episodes. The Unity environment will then pause, waiting for further signals telling it what to do next. Thus, not seeing any animation is expected when running this cell.
'''

for episode in range(1):
    env.reset()
    step_result = env.get_step_result(group_name)
    done = False
    episode_rewards = 0
    while not done:
        action_size = group_spec.action_size
        if group_spec.is_action_continuous():
            action = np.random.randn(step_result.n_agents(), group_spec.action_size)

        # if group_spec.is_action_discrete():
        #     branch_size = group_spec.discrete_action_branches
        #     action = np.column_stack([np.random.randint(0, branch_size[i], size=(step_result.n_agents())) for i in range(len(branch_size))])
        env.set_actions(group_name, action)
        env.step()
        step_result = env.get_step_result(group_name)
        step_result.obs[0].shape
        # step_result.get_agent_step_result(step_result.agent_id[0])
        step_result.get_agent_step_result(step_result.agent_id[0])
        episode_rewards += step_result.reward[0]
        done = step_result.done[0]
    print("Total reward this episode: {}".format(episode_rewards))

'''Close the environment when finished'''
'''
When we are finished using an environment, we can close it with the function below.
'''
env.close()
