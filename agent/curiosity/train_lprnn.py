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

    from learning_progress_rnn import GOALRNN
    net = GOALRNN(batch_size, number_layers, observation_dim, action_dim, goal_dim, n_hidden=256)

    # load rnn if saved. TODO: save only weights maybe to avoid bugs after we change architecture..
    if os.path.isfile("lprnn"+experiment_name+".pt"):
        temp_net = torch.load("lprnn"+experiment_name+".pt")
        net.load_state_dict(temp_net.state_dict())

    if os.path.isfile("lprnn_weights"+experiment_name+".pt"):
        net.load_state_dict(torch.load("lprnn_weights"+experiment_name+".pt"))

    # optimizer and losses
    from torch import optim
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    goal_loss = nn.MSELoss()
    goal_reconstruction_loss = nn.MSELoss()
    action_reconstruction_loss = nn.MSELoss()

    # initial values of several variables
    previous_goal_reward = torch.Tensor([-10])
    previous_lp_value = torch.zeros(1)
    learning_progress = torch.zeros(1)
    average_reward_estimate = 0
    average_lp_estimate = 0

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
        #goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
        #goal = np.reshape(goal.T,(-1))
        goal = torch.Tensor(pen_goal)

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

    for iteration in range(1000000):
        #print(observations)
        if evaluating: #if evaluating we just use the action prediction part of the network
            action_parameters = net.predict_action(goal,observations[0,0,:])
            #print(action_parameters.shape)
        else:
            #feed observations to net, get desired goal, actions (and their probabilities), and predicted value of action, and goal
            action, log_prob_action, goal, log_prob_goal, value, lp_value = net(observations, learning_progress.detach().unsqueeze(0).unsqueeze(0))
            goal = Variable(goal.data, requires_grad=True)
            if lp_training:
                action_parameters, log_prob_action, goal, log_prob_goal, value, lp_value = action[0,0,:], log_prob_action[0,0], goal[0,0,:], log_prob_goal[0,0], value[0,0,:], lp_value[0,0,:]
            else: # if we are not training goal policy then ignore the goal policy variables. We'll us the goal provided by openaigym
                action_parameters, log_prob_action, _, _, value, lp_value = action[0,0,:], log_prob_action[0,0], goal[0,0,:], log_prob_goal[0,0], value[0,0,:], lp_value[0,0,:]

        action_parameters = action_parameters.detach().numpy()
        #print(action_parameters)
        #run action using DMP
        for i in range(n_simulation_steps):
            # print(context.shape)
            action = dmp.action_rollout(None,action_parameters,i)
            results = env.step(action)
            if evaluating or not lp_training:
                pen_goal = results[0]["desired_goal"]
                #goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
                #goal = np.reshape(goal.T,(-1))
                goal = torch.Tensor(pen_goal)
                if evaluating:
                    print(results[1])
            if rendering:
                env.render()
            obs = results[0]["observation"]
            done = results[2]
            if done:
                print("reseting environment")
                results = env.reset()
                reset_env = True
                break
        new_observations = np.expand_dims(np.expand_dims(obs,0),0)
        new_observations = torch.Tensor(new_observations)

        if not evaluating: #if not evaluating, then train
            optimizer.zero_grad()

            # train policy to maximize goal reward (which is -goal_loss, which is -|observation-goal|^2) Here we are only looking at pen part of observation
            goal_reward = -goal_loss(new_observations[0,0,48:55],goal)
            print("goal_reward",goal_reward.data.item())
            rewards.append(goal_reward.data.item())

            hindsight_goal = new_observations[0,0,48:55]
            # we train for the goal reconstruction part of the network
            # we use the hindsight_goal (the outcome of our action, to ensure we autoencode reachable goals, and explore more effectively
            reconstructed_goal = net.autoencode_goal(hindsight_goal+0.01*torch.randn_like(hindsight_goal))
            loss = goal_reconstruction_loss(goal, reconstructed_goal)
            print("goal_reconstruction_loss", loss.data.item())
            partial_backprop(loss)

            # we also learn to predict the actions we just performed when goal is the observed outcome
            # this is called hindsight experience replay
            predicted_action_parameters = net.predict_action(hindsight_goal,observations[0,0,:])
            loss = action_reconstruction_loss(predicted_action_parameters, torch.Tensor(action_parameters))
            partial_backprop(loss, [net.goal_encoder])

            # we update the policy and value function following a non-bootstraped actor-critic approach
            # we update the state-action value function by computing delta,
            # delta is an unbiased estimator of the difference between the predicted value `value` and
            # the true expected reward (estimated by the observed `goal_reward`)
            # we train the value function to minimize their squared difference delta**2
            delta = goal_reward.detach() - value
            reward_value_fun = 0.5*delta**2
            partial_backprop(reward_value_fun,[net.goal_decoder])

            # then we update the policy using a policy gradient update
            # where delta is used as the advantage
            # note that we detach delta, so that it becomes a scalar, and gradients aren't backpropagated through it anymore
            loss_policy = delta.detach()*log_prob_action
            partial_backprop(loss_policy,[net.goal_decoder])

            # we define absolute learning progress as the absolute value of the "Bellman" error, `delta`
            # If delta is high, that means that the reward we got was significantly different from our expectations
            # which means we updated our policy a lot
            # which I am interpreting as "you have learned a lot" -- you have made significant learning progress
            # on the other hand if delta is very small, you didn't learn anything you didn't already know.
            learning_progress = torch.abs(delta)
            lps.append(learning_progress.data.item())

            # we use `learning_progress` (lp) as reward to train the goal-generating process (aka goal policy).
            # because the agent will be exploring goals in a continual way
            # we use a "continual learning" method for RL
            # in particular we use the average reward method, explained in Sutton and Barto 2nd edition (10.3, 13.6)
            # In short average reward RL uses differential value functions
            # which estimate the difference between {expected average reward following a state-action} and {average reward over all time - assume ergodicity}
            # -- called the differential return --
            # this difference measures the transient advantage of performing this action over other actions
            # there is a version of the policy gradient theorem which shows that using this in place of the expected raw reward in the episodic setting,
            # means we are maximizing the average reward over all time of our policy, which is the desired objective in the continual setting.
            # yeah, this theory has assumptions like ergodicity, and also you can only prove optimality for tabular RL, or linear models, not NNs,
            # but these are problems with all of RL theory really.
            # anyway, `delta` is now the Bellman error measuring the difference between
            # {the previous estimation of the expected differential return (`previous_lp_value`)}
            # and {the new bootstraped estimate `learning_progress.detach() - average_lp_estimate + lp_value.detach()`}
            delta = learning_progress.detach() - average_lp_estimate + lp_value.detach() - previous_lp_value
            # note that we detach the learning progress and lp_value, this is standard in Bellman errors I think
            # we don't backpropagate through the bootstraped target!
            # also update `average_lp_estimate` which is the estimate of the average reward.
            average_lp_estimate = average_lp_estimate + alpha*delta.detach()

            if iteration>0 and lp_training: #only do this once we have a previous_lp_value
                # update the differential value function for goal policy
                loss_lp_value_fun = 0.5*delta**2
                partial_backprop(loss_lp_value_fun,[net.goal_decoder])

                # update goal policy using policy gradient
                loss_goal_policy = delta.detach()*log_prob_goal
                # we don't update the goal_decoder because that way we are just training the RNN to produce certain action vectors
                # after the autoencoder has trained well, then each latent vector represents 1-to-1 a goal
                # and the action can learn to map to the actions corresponding to that goal
                # if we kept changing the goal_decoder, then the action decoder may "get confused" as its actual goal is changing even for fixed input latent vector
                partial_backprop(loss_goal_policy,[net.goal_decoder])
                optimizer.step()

            previous_lp_value = lp_value

            # saving rewards, learning progresses, etc
            if iteration % save_freq == save_freq -1:
                print("Saving stuff")
                #torch.save(net, "lprnn.pt")
                torch.save(net.state_dict(), "lprnn_weights"+experiment_name+".pt")
                with open("rewards"+experiment_name+".txt","a") as f:
                    f.write("\n".join([str(r) for r in rewards]))
                rewards = []
                with open("learning_progresses"+experiment_name+".txt","a") as f:
                    f.write("\n".join([str(lp) for lp in lps]))
                lps = []

            if save_goals:
                if iteration == 0:
                    goals = np.expand_dims(goal,0)
                else:
                    goals = np.concatenate([goals,np.expand_dims(goal,0)],axis=0)

        # we detach the RNN every so often, to determine how far to backpropagate through time
        # TODO: do this in a way that doesn't start from scratch, but instead backpropagates forget_freq many iterations in the past
        # at *every* time step!!
        if iteration % forget_freq == forget_freq -1:
            net.forget()


        # from rlpyt.envs.atari.atari_env import AtariEnv
        # AtariEnv("pong").action_space
        # import rlpyt
        # from rlpyt.algos.pg.a2c import A2C

from absl import app
if __name__ == '__main__':
    app.run(main)

#TODO
'''
more general goal specifier, which is just a parametrizer of reward given sequence of observations.
'''
