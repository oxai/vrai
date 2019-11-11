
'''

Ok, I want a neural network that takes a context, goal_space and goal, and produces an action parameter

The idea of the inverse model is to use an LSTM that is fed as hidden state the context, and is fed the temporal goal, time step by time step
and it has to predict the action parameters time step by time step too

'''

import tensorflow as tf

import tensorflow.keras as keras

# CuDNNLSTM

# layer = keras.layers.LSTM(1,return_sequences=True)

inputs = keras.layers.Input(shape=(n_steps+2,context_dim))
input_context = keras.layers.Input(shape=(context_dim,))

hidden_state_dim = 128
layer1 = keras.layers.GRU(hidden_state_dim,return_sequences=True)
layer2 = keras.layers.Dense(n_actuators)

# model = keras.Sequential([layer])

output1 = layer1(inputs)
output = layer2(output1)

model = keras.models.Model(inputs = inputs, outputs = output)

import numpy as np

# np.random.rand(1,10,10).astype(np.double).dtype

# output = model(np.random.rand(1,10,10).astype(np.float32))

import pickle
database = pickle.load(open("database.p","rb"))

# from sandboxv2 import outcome_dim

context_dim = 61
n_simulation_steps = 20
outcome_sampling_frequency=2
n_steps = n_simulation_steps//outcome_sampling_frequency
n_actuators = 20
n_dmp_basis = 10
# action_dim = n_steps*env.action_space.shape[0]
action_dim = n_actuators*(n_dmp_basis+1) # +1 for target position, in dmp parametrization
outcome_dim = context_dim*n_steps
memory_dim = context_dim+outcome_dim+action_dim
rendering = True
evaluating = False
# rendering = False
save_freq = 1000

contexts, outcomes, actions = database[:,:context_dim], database[:,context_dim:-action_dim], database[:,-action_dim:]

outcomes = outcomes.reshape((database.shape[0],context_dim,n_steps))
outcomes.shape
# outcomes = outcomes[:,:,:10]
outcomes = np.concatenate([outcomes,np.zeros(outcomes.shape[0:2]+(1,))],2)
contexts.shape
outcomes = np.concatenate([np.expand_dims(contexts,2),outcomes],2)
outcomes = outcomes.transpose(0,2,1)

actions.shape
actions = actions.reshape((-1,n_actuators,(n_dmp_basis+1)))
actions = actions.transpose(0,2,1)
actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[2])),actions],1)

model.compile("adam",loss="mse")

model.fit(outcomes,actions)
