#%%
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils import data
from torch.nn import functional as F

def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return Variable(h.data) if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)

import os,sys


# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_DIR = "."
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
AGENT_DIR = os.path.join(ROOT_DIR, 'agent')
MEMORIES_DIR = os.path.join(THIS_DIR,"memories")
# ENVIRONMENT_DIR = os.path.join(ROOT_DIR, 'environment')
# EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
sys.path.append(ROOT_DIR)
sys.path.append(AGENT_DIR)

#%%

from scipy.stats import multivariate_normal
#
# rv = multivariate_normal(mean=[0.5,0.0])
#
# rv.pdf([10,10])
#
# rv.rvs()
# rv = multivariate_normal(mean=np.zeros(goal_dim))

from torch.distributions.multivariate_normal import MultivariateNormal

# m = MultivariateNormal(torch.zeros(2), torch.eye(2))
#
# m.sample()
#
# m.log_prob(torch.zeros(2))

def MLP(input_dim, output_dim, number_layers=2, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = max(input_dim,output_dim)
    layers = sum(
        [[nn.Linear(input_dim, hidden_dim), nn.ReLU()]]
        + [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for i in range(number_layers-2)]
        + [[nn.Linear(hidden_dim, output_dim)]]
        , [])

    return nn.Sequential(*layers)

class GOALRNN(nn.Module):
    def __init__(self, bs, nl, observation_dim, action_dim, goal_dim, n_hidden = 128):
        super().__init__()
        self.nl = nl
        self.n_hidden = n_hidden
        # print(observation_dim+1)
        self.rnn = nn.LSTM(observation_dim+1, n_hidden, nl)
        self.goal_decoder = MLP(n_hidden, goal_dim)
        self.action_decoder = MLP(n_hidden+observation_dim, action_dim)
        self.value_decoder = MLP(n_hidden+observation_dim, 1)
        self.lp_decoder = MLP(n_hidden+observation_dim, 1)
        self.init_hidden(bs)

    def forward(self, observations, lps):
        bs = observations[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        # print(torch.cat([observations,lps],dim=2).shape)
        latents,h = self.rnn(torch.cat([observations,lps],dim=2), self.h) #(seq_len, batch, input_size) -> (seq_len, batch, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size):
        latent_and_observation = torch.cat([latents, observations], dim=2)
        goal_means = self.goal_decoder(latents)
        m = MultivariateNormal(goal_means, torch.eye(goal_dim))
        goals = m.sample()
        log_prob_goals = m.log_prob(goals)
        # goals = goal_means +
        # prob_goals = rv.pdf(goals)
        action_means = self.action_decoder(latent_and_observation)
        m = MultivariateNormal(action_means, torch.eye(action_dim))
        actions = m.sample()
        log_prob_actions = m.log_prob(action_means)
        values = self.value_decoder(latent_and_observation)
        lp_values = self.lp_decoder(latent_and_observation)
        # return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
        return actions, log_prob_actions, goals, log_prob_goals, values, lp_values

    def init_hidden(self, bs):
        self.h = (Variable(torch.zeros(self.nl, bs, self.n_hidden)),
                  Variable(torch.zeros(self.nl, bs, self.n_hidden)))

    def forget(self):
        self.h = repackage_var(self.h)



import gym
env=gym.make("HandManipulatePen-v0")
env.reset();
env.observation_space["observation"].shape[0]

#%%

observation_dim = env.observation_space["observation"].shape[0]
n_actuators = env.action_space.shape[0]
n_dmp_basis = 10
action_dim = n_actuators*(n_dmp_basis+1)
#goal_dim = observation_dim
goal_dim = 7

batch_size = 1
number_layers = 2
print(observation_dim)
net = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim)

# net.rnn

from torch import optim
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

goal_loss = nn.MSELoss()

action = env.action_space.sample()
observation = env.step(action)[0]["observation"]
observations = np.expand_dims(np.expand_dims(observation,0),0)
observations = torch.Tensor(observations)

previous_goal_reward = torch.Tensor([-10])
# previous_value = torch.zeros(1)
previous_lp_value = torch.zeros(1)
learning_progress = torch.zeros(1)
average_reward_estimate = 0
average_lp_estimate = 0
alpha = 0.1

env.relative_control = True

from dmp import DMP
n_simulation_steps=25
dmp = DMP(10,n_simulation_steps,n_actuators)

#%%

save_goals = False
#rendering = True
rendering = False
save_freq = 300
forget_freq = 300
# forget_freq = 2

if os.path.isfile("lprnn.pt"):
    net = torch.load("lprnn.pt")

rewards = []
lps = []

for iteration in range(1000000):

    action, log_prob_action, goal, log_prob_goal, value, lp_value = net(observations, learning_progress.detach().unsqueeze(0).unsqueeze(0))
    goal = Variable(goal.data, requires_grad=True)
    action_parameters, log_prob_action, goal, log_prob_goal, value, lp_value = action[0,0,:], log_prob_action[0,0], goal[0,0,:], log_prob_goal[0,0], value[0,0,:], lp_value[0,0,:]
    action_parameters = action_parameters.detach().numpy()
    for i in range(n_simulation_steps):
        # print(context.shape)
        action = dmp.action_rollout(None,action_parameters,i)
        results = env.step(action)
        if rendering:
            env.render()
        obs = results[0]["observation"]
        # print(obs)
        done = results[2]
        if done:
            print("reseting environment")
            results = env.reset()
            reset_env = True
            break
        # if i % outcome_sampling_frequency == 0:
        #     observations.append(obs)
    # results = env.step(action)
    # observation = results[0]["observation"]
    # done = results[2]
    # env.render()
    # if done:
    #     print("reseting environment")
    #     results = env.reset()
    #     # obs = results["observation"]
    #     reset_env = True
    #     continue
    # if reset_env:
    #     reset_env = False
    #     continue
    observations = np.expand_dims(np.expand_dims(obs,0),0)
    observations = torch.Tensor(observations)

    def partial_backprop(loss,parts_to_ignore):
        for part in parts_to_ignore:
            for parameter in part.parameters():
                parameter.requires_grad = False
        loss.backward(retain_graph=True)
        for part in parts_to_ignore:
            for parameter in part.parameters():
                parameter.requires_grad = True

    optimizer.zero_grad()
    goal_reward = -goal_loss(observations[0,0,48:55],goal)
    print("goal_reward",goal_reward.data.item())
    rewards.append(goal_reward.data.item())

    # delta = goal_reward - average_reward_estimate + value.detach() - previous_value
    delta = goal_reward.detach() - value
    # average_reward_estimate = average_reward_estimate + alpha*delta.detach()
    reward_value_fun = 0.5*delta**2
    partial_backprop(reward_value_fun,[net.goal_decoder])

    loss_policy = delta.detach()*log_prob_action
    partial_backprop(loss_policy,[net.goal_decoder])

    # learning_progress = nn.ReLU()(goal_reward-previous_goal_reward)
    # learning_progress = torch.abs(goal_reward-previous_goal_reward)
    learning_progress = torch.abs(delta)
    lps.append(learning_progress.data.item())


    delta = learning_progress.detach() - average_lp_estimate + lp_value.detach() - previous_lp_value
    average_lp_estimate = average_lp_estimate + alpha*delta.detach()

    if iteration>0:
        loss_lp_value_fun = 0.5*delta**2
        partial_backprop(loss_lp_value_fun,[net.goal_decoder])

    loss_goal_policy = delta.detach()*log_prob_goal
    partial_backprop(loss_goal_policy,[net.goal_decoder])
    optimizer.step()

    # previous_goal_reward = goal_reward
    previous_lp_value = lp_value
    # previous_value = value

    if iteration % save_freq == save_freq -1:
        print("Saving stuff")
        torch.save(net, "lprnn.pt")
        with open("rewards.txt","a") as f:
            f.write("\n".join([str(r) for r in rewards]))
        rewards = []
        with open("learning_progresses.txt","a") as f:
            f.write("\n".join([str(lp) for lp in lps]))
        lps = []

    if save_goals:
        if iteration == 0:
            goals = np.expand_dims(goal,0)
        else:
            goals = np.concatenate([goals,np.expand_dims(goal,0)],axis=0)

    if iteration % forget_freq == forget_freq -1:
        net.forget()


    # from rlpyt.envs.atari.atari_env import AtariEnv
    # AtariEnv("pong").action_space
    # import rlpyt
    # from rlpyt.algos.pg.a2c import A2C

#TODO
'''
more general goal specifier, which is just a parametrizer of reward given sequence of observations.
'''
