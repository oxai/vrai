import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import os,sys
THIS_DIR = "."
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
AGENT_DIR = os.path.join(ROOT_DIR, 'agent')
MEMORIES_DIR = os.path.join(THIS_DIR,"memories")
sys.path.append(ROOT_DIR)
sys.path.append(AGENT_DIR)

# from agent.robot_hand_utils import render_with_target, setup_render
from agent.robot_fetch_utils import render_with_target, setup_render

#%%
# training/eval arguments
from absl import flags

flags.DEFINE_integer("save_freq",300,"frequency at which we save stuff: model, rewards, LPs")
flags.DEFINE_integer("forget_freq",100,"frequency at which we reset the backpropagation thru time. TODO: fix this")
flags.DEFINE_bool("rendering",False,"whether to render the environment or not")
flags.DEFINE_bool("evaluating",False,"whether we are evaluating on the target goals, or still learning/exploring")
flags.DEFINE_bool("save_goals",False,"whether to save goals")
flags.DEFINE_string("experiment_name","","experiment name")
flags.DEFINE_bool("lp_training",True,"whether to train the goal policy or use random goals")

# FLAGS = {}
# FLAGS["save_freq"] = 300
# FLAGS["forget_freq"] = 100
# FLAGS["rendering"] = False
# FLAGS["evaluating"] = False
# FLAGS["save_goals"] = False
# FLAGS["experiment_name"] = ""
# FLAGS["lp_training"] = True

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def main(argv):
    FLAGS = flags.FLAGS
    print(FLAGS.flag_values_dict())
    FLAGS = FLAGS.flag_values_dict()
    globals().update(FLAGS)

    '''ENVIRONMENT'''
    import gym
    # env=gym.make("HandManipulatePen-v0")
    env=gym.make("FetchReach-v1")
    results = env.reset();
    #env.observation_space["observation"].shape[0]
    observation_dim = env.observation_space["observation"].shape[0]
    n_actuators = env.action_space.shape[0]
    n_dmp_basis = 10
    # action_dim = n_actuators*(n_dmp_basis+1)
    action_dim = n_actuators
    #goal_dim = observation_dim
    goal_dim = results["desired_goal"].shape[0]
    batch_size = 1
    number_layers = 2
    gamma=0.9
    #alpha = 0.1 # hyperparameter used in average lp estimate for goal policy

    #DMP
    env.relative_control = True
    from dmp import DMP
    # n_simulation_steps=25
    n_simulation_steps=1
    dmp = DMP(10,n_simulation_steps,n_actuators)

    #%%

    '''NET'''

    from learning_progress_ddpg import GOALRNN
    net = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim, n_hidden=256)
    net2 = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim, n_hidden=256)
    net2.load_state_dict(net.state_dict(),strict=False)

    if os.path.isfile("ddpg_weights"+experiment_name+".pt"):
        print("LOADING WEIGHTS")
        net.load_state_dict(torch.load("ddpg_weights"+experiment_name+".pt"))

    #print(net.goal_decoder.state_dict().items())
    #net.goal_decoder.apply(weight_reset)
    #print(net.goal_decoder.state_dict().items())

    # optimizer and losses
    from torch import optim
    #optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    #optimizer = optim.RMSprop(net.parameters())

    # initial values of several variables
    learning_progress = torch.zeros(1)

    #initial run
    action = env.action_space.sample()
    results = env.step(action)
    observation = results[0]["observation"]
    observations = np.expand_dims(np.expand_dims(observation,0),0)
    observations = torch.Tensor(observations)

    train_freq = 250
    update_copy_freq = 250

    pen_goal = results[0]["desired_goal"]
    goal = torch.Tensor(pen_goal)
    if rendering:
        setup_render(env)

    rewards = []
    lps = []
    memory_buffer = []
    # a function that allows to do back prop, but not accumulate gradient in certain modules of a network
    def partial_backprop(loss,parts_to_ignore=[]):
        for part in parts_to_ignore:
            for parameter in part.parameters():
                parameter.requires_grad = False
        loss.backward(retain_graph=True)
        for part in parts_to_ignore:
            for parameter in part.parameters():
                parameter.requires_grad = True

    print(observations.shape)
    reset_env = False
    '''TRAINING LOOP'''
    # using DDPG on-olicy (without memory buffer for now. TODO: have memory buffer
    episode_number = 0
    for iteration in range(1000000):
        noisy_actions = net.compute_noisy_actions(goal.unsqueeze(0).unsqueeze(0),observations)
        action_parameters = noisy_actions[0,0,:]
        action_parameters = action_parameters.detach().numpy()
        for i in range(n_simulation_steps):
            # action = dmp.action_rollout(None,action_parameters,i)
            action = action_parameters
            results = env.step(action)
            pen_goal = results[0]["desired_goal"]
            goal = torch.Tensor(pen_goal)
            # print(goal)
            if rendering:
                render_with_target(env,goal.detach().numpy())
            obs = results[0]["observation"]
            achieved_goal = results[0]["achieved_goal"]
            done = results[2]
            sparse_reward = env.compute_reward(achieved_goal, goal.numpy(), None)
            reward = np.clip(-np.linalg.norm(achieved_goal - goal.numpy()), -1, 1)
            print(sparse_reward)
            if done:
                episode_number += 1
                print("num episodes", episode_number)
                # print("reseting environment")
                results = env.reset()
                obs = results["observation"]
                reset_env = True
                break
        new_observations = np.expand_dims(np.expand_dims(obs,0),0)
        new_observations = torch.Tensor(new_observations)
        achieved_goals = np.expand_dims(np.expand_dims(achieved_goal,0),0)
        achieved_goals = torch.Tensor(achieved_goals)

        if reset_env:
            observations = new_observations
            if not evaluating and lp_training:
                learning_progress = Variable(torch.zeros_like(learning_progress))
            reset_env = False
            # continue

        goals = goal.unsqueeze(0).unsqueeze(0)
        memory_buffer.append((observations, goals, noisy_actions.detach(), achieved_goals, reward, new_observations))

        if len(memory_buffer) > 50000:
            memory_buffer = []

        if iteration % update_copy_freq == update_copy_freq-1 :
            net2.load_state_dict(net.state_dict(),strict=False)
        if iteration % train_freq == train_freq-1:
            print("Training")
            for i in range(100):
                index = np.random.choice(range(len(memory_buffer)))
                # print(index)
                observations, goals, noisy_actions, hindsight_goals, reward, new_observations = memory_buffer[index]
                optimizer.zero_grad()
                value = net.compute_q_value(goals, observations, noisy_actions)
                # print("q-value", value.data.item())
                delta = reward + gamma*net2.compute_q_value(goals,new_observations,net2.compute_actions(goals,new_observations))- value
                reward_value_fun = 0.5*delta**2
                partial_backprop(reward_value_fun)
                optimizer.step()

            for i in range(100):
                index = np.random.choice(range(len(memory_buffer)))
                # print(index)
                observations, goals, noisy_actions, hindsight_goals, reward, new_observations = memory_buffer[index]
                optimizer.zero_grad()
                loss_policy = -net.compute_q_value(goals, observations, noisy_actions)
                partial_backprop(loss_policy)
                optimizer.step()


        if iteration % save_freq == save_freq -1:
            # print("Saving stuff")
            torch.save(net.state_dict(), "ddpg_weights"+experiment_name+".pt")

        observations = new_observations

from absl import app
if __name__ == '__main__':
    app.run(main)

#TODO
'''
more general goal specifier, which is just a parametrizer of reward given sequence of observations.
'''
