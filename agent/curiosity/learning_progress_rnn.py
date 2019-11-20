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
    return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)

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

class GOALRNN(nn.Module):
    def __init__(self, bs, nl, observation_dim, action_dim, goal_dim, n_hidden = 128):
        super().__init__()
        self.nl = nl
        self.n_hidden = n_hidden
        self.rnn = nn.LSTM(observation_dim, n_hidden, nl)
        self.goal_decoder = nn.Linear(n_hidden, goal_dim)
        self.action_decoder = nn.Linear(n_hidden+observation_dim, action_dim)
        self.value_decoder = nn.Linear(n_hidden+observation_dim, 1)
        self.lp_decoder = nn.Linear(n_hidden+observation_dim, 1)
        self.init_hidden(bs)

    def forward(self, observations):
        bs = observations[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        latents,h = self.rnn(observations, self.h) #(seq_len, batch, input_size) -> (seq_len, batch, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size):
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
        self.h = repackage_var(h)



import gym
env=gym.make("HandManipulatePen-v0")
env.reset()
env.observation_space["observation"].shape[0]

#%%

observation_dim = env.observation_space["observation"].shape[0]
n_actuators = env.action_space.shape[0]
n_dmp_basis = 10
action_dim = n_actuators*(n_dmp_basis+1)
goal_dim = observation_dim

batch_size = 1
number_layers = 2
net = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim)

from torch import optim
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

goal_loss = nn.MSELoss()

action = env.action_space.sample()
observation = env.step(action)[0]["observation"]
observations = np.expand_dims(np.expand_dims(observation,0),0)
observations = torch.Tensor(observations)

previous_goal_reward = torch.Tensor([-10])
previous_value = torch.zeros(1)
previous_lp_value = torch.zeros(1)
average_reward_estimate = 0
average_lp_estimate = 0
alpha = 0.1

env.relative_control = True

from dmp import DMP

dmp = DMP(10,n_simulation_steps,n_actuators)

#%%

n_simulation_steps=25
rendering = True

for itertion in range(10000):

    action, log_prob_action, goal, log_prob_goal, value, lp_value = net(observations)
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
    goal_reward = goal_loss(observations,goal)

    delta = goal_reward - average_reward_estimate + value.detach() - previous_value
    average_reward_estimate = average_reward_estimate + alpha*delta.detach()
    loss_value_fun = 0.5*delta**2
    partial_backprop(loss_value_fun,[net.goal_decoder,net.value_decoder])

    loss_goal_policy = delta.detach()*log_prob_action
    partial_backprop(loss_goal_policy,[net.goal_decoder,net.value_decoder])

    learning_progress = nn.ReLU()(goal_reward-previous_goal_reward)

    delta = learning_progress - average_reward_estimate + value.detach() - previous_value
    average_reward_estimate = average_reward_estimate + alpha*delta.detach()

    loss_value_fun = 0.5*delta**2
    partial_backprop(loss_value_fun,[net.goal_decoder,net.action_decoder])

    loss_goal_policy = delta.detach()*log_prob_goal
    partial_backprop(loss_goal_policy,[net.goal_decoder,net.value_decoder,net.action_decoder])
    optimizer.step()

    previous_goal_reward = goal_reward
    previous_value = value

    # from rlpyt.envs.atari.atari_env import AtariEnv
    # AtariEnv("pong").action_space
    # import rlpyt
    # from rlpyt.algos.pg.a2c import A2C
