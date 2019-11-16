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

class CharSeqStatefulLSTM(nn.Module):
    def __init__(self, bs, nl, observation_dim, action_dim, goal_dim, n_hidden = 128):
        super().__init__()
        self.nl = nl
        self.n_hidden = n_hidden
        self.rnn = nn.LSTM(observation_dim, n_hidden, nl)
        self.goal_decoder = nn.Linear(n_hidden, goal_dim)
        self.action_decoder = nn.Linear(n_hidden+observation_dim, action_dim)
        self.value_decoder = nn.Linear(n_hidden+observation_dim, action_dim)
        self.init_hidden(bs)

    def forward(self, observations):
        bs = observations[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        latents,h = self.rnn(observations, self.h) #(seq_len, batch, input_size) -> (seq_len, batch, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size):
        action_observation = torch.cat([latents, observations], dim=2)
        goals = self.goal_decoder(latents)
        actions = self.action_decoder(action_observation)
        values = self.value_decoder(action_observation)
        # return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
        return actions, goals, values

    def init_hidden(self, bs):
        self.h = (Variable(torch.zeros(self.nl, bs, self.n_hidden)),
                  Variable(torch.zeros(self.nl, bs, self.n_hidden)))

    def forget(self):
        self.h = repackage_var(h)



import gym
env=gym.make("HandManipulatePen-v0")
env.reset()
env.observation_space["observation"].shape[0]


observation_dim = env.observation_space["observation"].shape[0]
action_dim = env.action_space.shape[0]
goal_dim = observation_dim

batch_size = 1
number_layers = 2
net = CharSeqStatefulLSTM(batch_size, number_layers, observation_dim, action_dim, goal_dim)

optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

goal_loss = nn.MSELoss()

action = env.action_space.sample()
observation = env.step(action)[0]["observation"]
observations = np.expand_dims(np.expand_dims(observation,0),0)
observations = torch.Tensor(observations)

previous_goal_reward = torch.Tensor([-10])
previous_value = torch.zeros(1)
average_reward_estimate = 0
alpha = 0.1

action_full, goal_full, value_full = net(observations)
action, goal, value = action_full[0,0,:], goal_full[0,0,:],  value_full[0,0,:]
observation = env.step(action.detach().numpy())[0]["observation"]
observations = np.expand_dims(np.expand_dims(observation,0),0)
observations = torch.Tensor(observations)

goal_reward = -goal_loss(observations,goal_full)

optimizer.zero_grad()
for parameter in net.goal_decoder.parameters():
    parameter.requires_grad = False

for parameter in net.value_decoder.parameters():
    parameter.requires_grad = False

goal_reward.backward()
for parameter in net.goal_decoder.parameters():
    parameter.requires_grad = True

for parameter in net.value_decoder.parameters():
    parameter.requires_grad = True

learning_progress = nn.ReLU(goal_reward-previous_goal_reward)

delta = learning_progress - average_reward_estimate + value - previous_value
average_reward_estimate = average_reward_estimate + alpha*delta

for parameter in net.action_decoder.parameters():
    parameter.requires_grad = False

for parameter in net.goal_decoder.parameters():
    parameter.requires_grad = False

loss_value_fun = 0.5*delta**2
loss_value_fun.backward()
optimizer.step()

for parameter in net.action_decoder.parameters():
    parameter.requires_grad = True

for parameter in net.goal_decoder.parameters():
    parameter.requires_grad = True

for parameter in net.action_decoder.parameters():
    parameter.requires_grad = False

for parameter in net.value_decoder.parameters():
    parameter.requires_grad = False

loss_goal_policy = delta*np.log(pi(action))
loss_goal_policy.backward()
optimizer.step()

for parameter in net.action_decoder.parameters():
    parameter.requires_grad = True

for parameter in net.value_decoder.parameters():
    parameter.requires_grad = True

previous_goal_reward = goal_reward
previous_value = value

# from rlpyt.envs.atari.atari_env import AtariEnv
# AtariEnv("pong").action_space
# import rlpyt
# from rlpyt.algos.pg.a2c import A2C
