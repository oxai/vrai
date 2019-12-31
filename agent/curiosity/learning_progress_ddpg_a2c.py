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
        + [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] for i in range(number_layers-2)]
        #+ [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for i in range(number_layers-2)]
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
        self.goal_decoder = MLP(observation_dim, goal_dim*2)
        self.action_decoder = MLP(goal_dim+observation_dim, action_dim, hidden_dim=1024)
        self.q_value_decoder = MLP(goal_dim+observation_dim+action_dim, 1, number_layers=1, hidden_dim=1024)
        self.vlp_decoder = MLP(observation_dim, 1)
        self.pen_vars_slice = slice(54,61)

    def forward(self, observations):
        pen_pos_center = torch.Tensor([1.0,0.90,0.15]).unsqueeze(0).unsqueeze(0)

        noisy_goals, log_prob_goals = self.compute_noisy_goals(observations)
        #print(observations,noisy_goals)
        actions = self.compute_actions(noisy_goals, observations)
        state_dict_backup = self.action_decoder.state_dict()
        with torch.no_grad():
            for param in self.action_decoder.parameters():
                param.add_(torch.randn(param.size()) * 0.8)
        noisy_actions = self.compute_actions(noisy_goals, observations)
        self.action_decoder.load_state_dict(state_dict_backup, strict=False)
        #print(actions)
        #if np.random.rand() < 0.2:
        #    noisy_actions = actions + 5*torch.randn_like(actions)
        #else:
        #    noisy_actions = actions + np.random.randn()*torch.randn_like(actions)
        #noisy_actions = actions
        #values = self.compute_q_value(noisy_goals, observations, noisy_actions)
        #lp_values = self.compute_qlp(observations, noisy_goals)
        return actions, noisy_actions, noisy_goals, log_prob_goals#, values, lp_values

    
    def compute_goals(self,observations):
        pen_vars_slice = self.pen_vars_slice
        goal_means, goal_stds = torch.split(self.goal_decoder(observations), self.goal_dim, dim=2)
        m = MultivariateNormal(goal_means, (goal_stds**2+0.01)*torch.eye(self.goal_dim)) # squaring stds so as to be positive
        goals = m.sample()
        log_prob_goals = m.log_prob(goals)
        goals = torch.tanh(goals)
        pen_pos = observations[:,:,pen_vars_slice][...,:3]
        pen_rot = observations[:,:,pen_vars_slice][...,3:]
        rot_goal = goals[:,:,3:]
        #rel_rot_goal = rot_goal*0.1+pen_rot
        rel_rot_goal = rot_goal+pen_rot
        goals = torch.cat([(goals[:,:,:3])*0.1+pen_pos,(rel_rot_goal)/torch.norm(rel_rot_goal)], dim=2)
        return goals, log_prob_goals

    def compute_log_prob_goals(self,observations, goals):
        pen_vars_slice = self.pen_vars_slice
        goal_means, goal_stds = torch.split(self.goal_decoder(observations), self.goal_dim, dim=2)
        m = MultivariateNormal(goal_means, (goal_stds**2+0.01)*torch.eye(self.goal_dim)) # squaring stds so as to be positive
        log_prob_goals = m.log_prob(goals)
        return log_prob_goals
    
    def compute_noisy_goals(self,observations):
        state_dict_backup = self.goal_decoder.state_dict()
        with torch.no_grad():
            for param in self.goal_decoder.parameters():
                param.add_(torch.randn(param.size()) * 0.1)
        noisy_goals, log_prob_goals = self.compute_goals(observations)
        noisy_goals = torch.clamp(noisy_goals, -10, 10)
        self.goal_decoder.load_state_dict(state_dict_backup, strict=False)
        #goals = self.compute_goals(observations)
        #if np.random.rand() < 0.2:
        #    noisy_goals = goals + torch.Tensor([0.05]*3+[1.0]*4)*torch.randn_like(goals)
        #else:
        #    noisy_goals = goals + 0.001*torch.randn_like(goals)
        return noisy_goals, log_prob_goals

    def compute_q_value(self, goals, observations, actions):
        values = self.q_value_decoder(torch.cat([goals,observations,actions], dim=2))
        values = values + 1 # learn the difference between the value and -1, because at the beginning most values will be close to -1
        #values = torch.tanh(values)
        return values

    def compute_vlp(self, observations):
        values = self.vlp_decoder(observations)
        #values = values - 1 #pesimistic goal values to see if it makes it not explode :P
        #values = torch.tanh(values)
        return values

    def compute_actions(self, goals, observations):
        actions = self.action_decoder(torch.cat([goals, observations], dim=2))
        #actions = torch.tanh(actions)
        #print(actions)
        return actions
