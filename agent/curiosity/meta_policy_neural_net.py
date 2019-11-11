import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

model.compile("adam",loss="mse")

inputs = keras.layers.Input(shape=(n_steps+2,context_dim))
input_context = keras.layers.Input(shape=(context_dim,))

hidden_state_dim = 128
layer1 = keras.layers.GRU(hidden_state_dim,return_sequences=True)
layer2 = keras.layers.Dense(n_actuators)

# model = keras.Sequential([layer])

output1 = layer1(inputs)
output = layer2(output1)

model = keras.models.Model(inputs = inputs, outputs = output)

def learn_from_database(database, FLAGS):
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

    model.fit(outcomes,actions)
