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
    def __init__(self, bs, nl, observation_dim, action_dim, goal_dim, n_hidden = 64, n_hidden2 = 64):
        super().__init__()
        self.nl = nl
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.goal_decoder = MLP(observation_dim, goal_dim)
        self.action_decoder = MLP(goal_dim+observation_dim, action_dim, hidden_dim=1024)
        self.q_value_decoder = MLP(goal_dim+observation_dim+action_dim, 1, hidden_dim=1024)
        self.qlp_decoder = MLP(observation_dim+goal_dim, 1)

    def forward(self, observations):
        goals = self.goal_decoder(observations)
        actions = self.compute_actions(goals, observations)
        values = self.compute_q_value(goals, observations, actions)
        lp_values = self.compute_qlp(observations, goals)
        return actions, goals, values, lp_values

    def compute_q_value(self, goals, observations, actions):
        values = self.q_value_decoder(torch.cat([goals,observations,actions], dim=2))
        return values

    def compute_qlp(self, observations, goals):
        values = self.qlp_decoder(torch.cat([observations,goals], dim=2))
        return values

    def compute_actions(self, goals, observations):
        actions = self.action_decoder(torch.cat([goals, observations], dim=2))
        return actions
