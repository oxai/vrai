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

from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

#Multi-layer perceptron
def MLP(input_dim, output_dim, number_layers=2, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = max(input_dim,output_dim)
    layers = sum(
        [[nn.Linear(input_dim, hidden_dim), nn.ReLU()]]
        + [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for i in range(number_layers-2)]
        + [[nn.Linear(hidden_dim, output_dim)]]
        , [])

    return nn.Sequential(*layers)

'''GOALRNN v1'''
class GOALRNN(nn.Module):
    def __init__(self, bs, nl, observation_dim, action_dim, goal_dim, n_hidden = 128):
        super().__init__()
        self.nl = nl
        self.n_hidden = n_hidden
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.rnn = nn.LSTM(observation_dim+1, n_hidden, nl)
        self.goal_decoder = MLP(n_hidden, goal_dim)
        self.goal_encoder = MLP(goal_dim, n_hidden)
        self.action_decoder = MLP(n_hidden+observation_dim, action_dim)
        self.value_decoder = MLP(n_hidden+observation_dim, 1)
        self.lp_decoder = MLP(n_hidden+observation_dim, 1)
        self.init_hidden(bs)

    def forward(self, observations, lps):
        bs = observations[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        latents,h = self.rnn(torch.cat([observations,lps],dim=2), self.h) #(seq_len, batch, input_size) -> (seq_len, batch, num_directions * hidden_size), (num_layers * num_directions, batch, hidden_size):
        latent_and_observation = torch.cat([latents, observations], dim=2)
        goal_means = self.goal_decoder(latents)
        m = MultivariateNormal(goal_means, torch.eye(self.goal_dim))
        goals = m.sample()
        log_prob_goals = m.log_prob(goals)
        action_means = self.action_decoder(latent_and_observation)
        m = MultivariateNormal(action_means, torch.eye(self.action_dim))
        actions = m.sample()
        log_prob_actions = m.log_prob(action_means)
        values = self.value_decoder(latent_and_observation)
        lp_values = self.lp_decoder(latent_and_observation)
        return actions, log_prob_actions, goals, log_prob_goals, values, lp_values

    def autoencode_goal(self, goal):
        latent = self.goal_encoder(goal)
        return self.goal_decoder(latent)

    def predict_action(self, goal):
        latent = self.goal_encoder(goal)
        return self.action_decoder(latent)

    def init_hidden(self, bs):
        self.h = (Variable(torch.zeros(self.nl, bs, self.n_hidden)),
                  Variable(torch.zeros(self.nl, bs, self.n_hidden)))

    def forget(self):
        self.h = repackage_var(self.h)
