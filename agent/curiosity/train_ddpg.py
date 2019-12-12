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

from agent.robot_hand_utils import render_with_target, setup_render

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

def main(argv):
    FLAGS = flags.FLAGS
    print(FLAGS.flag_values_dict())
    FLAGS = FLAGS.flag_values_dict()
    globals().update(FLAGS)

    '''ENVIRONMENT'''
    import gym
    env=gym.make("HandManipulatePen-v0")
    results = env.reset();
    env.observation_space["observation"].shape[0]
    observation_dim = env.observation_space["observation"].shape[0]
    n_actuators = env.action_space.shape[0]
    n_dmp_basis = 10
    action_dim = n_actuators*(n_dmp_basis+1)
    #goal_dim = observation_dim
    goal_dim = 7
    batch_size = 1
    number_layers = 2
    alpha = 0.1 # hyperparameter used in average lp estimate for goal policy

    #DMP
    env.relative_control = True
    from dmp import DMP
    n_simulation_steps=25
    dmp = DMP(10,n_simulation_steps,n_actuators)

    #%%

    '''NET'''

    from learning_progress_ddpg import GOALRNN
    net = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim, n_hidden=256)

    if os.path.isfile("lprnn_weights"+experiment_name+".pt"):
        print("LOADING WEIGHTS")
        net.load_state_dict(torch.load("lprnn_weights"+experiment_name+".pt"))

    # optimizer and losses
    from torch import optim
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    #optimizer = optim.RMSprop(net.parameters())
    goal_loss = nn.MSELoss()
    goal_reconstruction_loss = nn.MSELoss()
    action_reconstruction_loss = nn.MSELoss()

    # initial values of several variables
    previous_goal_reward = torch.Tensor([-1.0])
    previous_lp_value = torch.zeros(1)
    learning_progress = torch.zeros(1)

    #initial run
    action = env.action_space.sample()
    results = env.step(action)
    observation = results[0]["observation"]
    observations = np.expand_dims(np.expand_dims(observation,0),0)
    observations = torch.Tensor(observations)

    if evaluating:
        net.eval()
    if evaluating or not lp_training:
        pen_goal = results[0]["desired_goal"]
        goal = torch.Tensor(pen_goal)
    if rendering:
        setup_render(env)

    rewards = []
    lps = []
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
    for iteration in range(1000000):
        if evaluating: #if evaluating we just use the action prediction part of the network
            action_parameters,_ = net.compute_actions(goal.unsqueeze(0).unsqueeze(0),observations)
            action_parameters = action_parameters[0,0,:]
        else:
            #feed observations to net, get desired goal, actions (and their probabilities), and predicted value of action, and goal
            actions, noisy_actions, goals, noisy_goals, values, lp_values = net(observations)
            #goals = Variable(goals.data, requires_grad=True)
            if lp_training:
                action_parameters, goal, value, lp_value = noisy_actions[0,0,:], noisy_goals[0,0,:], values[0,0,:], lp_values[0,0,:]
            else: # if we are not training goal policy then ignore the goal policy variables. We'll us the goal provided by openaigym
                #action_parameters, log_prob_action, _, _, value, lp_value = actions[0,0,:], log_prob_action[0,0], goal[0,0,:], log_prob_goal[0,0], value[0,0,:], lp_value[0,0,:]
                pass

        action_parameters = action_parameters.detach().numpy()
        #run action using DMP
        for i in range(n_simulation_steps):
            action = dmp.action_rollout(None,action_parameters,i)
            results = env.step(action)
            if evaluating or not lp_training:
                pen_goal = results[0]["desired_goal"]
                goal = torch.Tensor(pen_goal)
                if evaluating:
                    print(results[1])
                    rewards.append(results[1])
            if rendering:
                render_with_target(env,goal.detach().numpy())
            obs = results[0]["observation"]
            done = results[2]
            if done:
                print("reseting environment")
                results = env.reset()
                obs = results["observation"]
                reset_env = True
                break
        new_observations = np.expand_dims(np.expand_dims(obs,0),0)
        new_observations = torch.Tensor(new_observations)

        if not evaluating:
            # saving rewards, learning progresses, etc
            if iteration % save_freq == save_freq -1:
                print("Saving stuff")
                torch.save(net.state_dict(), "lprnn_weights"+experiment_name+".pt")
                with open("rewards"+experiment_name+".txt","a") as f:
                    f.write("\n".join([str(r) for r in rewards]))
                    f.write("\n")
                rewards = []
                with open("learning_progresses"+experiment_name+".txt","a") as f:
                    f.write("\n".join([str(lp) for lp in lps]))
                    f.write("\n")
                lps = []

            if save_goals:
                if iteration == 0:
                    goals = np.expand_dims(goal,0)
                else:
                    goals = np.concatenate([goals,np.expand_dims(goal,0)],axis=0)
        else:
            if iteration % save_freq == save_freq -1:
                with open("test_rewards"+experiment_name+".txt","a") as f:
                    f.write("\n".join([str(r) for r in rewards]))
                rewards = []

        if reset_env:
            observations = new_observations
            if not evaluating and lp_training:
                previous_lp_value = lp_value
                learning_progress = Variable(torch.zeros_like(learning_progress))
            reset_env = False
            continue

        if not evaluating: #if not evaluating, then train
            optimizer.zero_grad()

            pen_vars_slice = slice(54,61)
            goal_reward = env.compute_reward(new_observations[0,0,pen_vars_slice].numpy(), goal.detach().numpy(), None)
            print(new_observations[0,0,pen_vars_slice], goal)
            print("goal_reward",goal_reward)
            rewards.append(goal_reward)

            delta = goal_reward - value
            print("q-value", value.data.item())
            reward_value_fun = 0.5*delta**2
            partial_backprop(reward_value_fun, [net.goal_decoder, net.action_decoder])

            loss_policy = -net.compute_q_value(noisy_goals.detach(), observations, actions)
            partial_backprop(loss_policy, [net.q_value_decoder])

            learning_progress = torch.abs(delta)
            lps.append(learning_progress.data.item())

            delta = learning_progress.detach() + lp_value.detach() - previous_lp_value

            if iteration>0 and lp_training: #only do this once we have a previous_lp_value
                loss_lp_value_fun = 0.5*delta**2
                partial_backprop(loss_lp_value_fun, [net.goal_decoder])

                loss_goal_policy = -net.compute_qlp(observations, goals)
                partial_backprop(loss_goal_policy, [net.qlp_decoder])

            optimizer.step()

            previous_lp_value = lp_value

        observations = new_observations

from absl import app
if __name__ == '__main__':
    app.run(main)

#TODO
'''
more general goal specifier, which is just a parametrizer of reward given sequence of observations.
'''
