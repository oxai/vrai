import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

num_gpus = 1
config = tf.compat.v1.ConfigProto()
if num_gpus > 0:
    config.gpu_options.allow_growth = True

tf.compat.v1.enable_eager_execution(config=config)
##the code below is necessary for keras not to use all memory
set_session = tf.compat.v1.keras.backend.set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def make_meta_policy_nn_model(FLAGS):
    globals().update(FLAGS)
    inputs = keras.layers.Input(shape=(n_steps+2,context_dim),dtype="float32")
    input_context = keras.layers.Input(shape=(context_dim,),dtype="float32")

    hidden_state_dim = 64
    layer1 = keras.layers.GRU(hidden_state_dim,return_sequences=True)
    layer2 = keras.layers.Dense(n_actuators)

    output1 = layer1(inputs)
    output = layer2(output1)

    model = keras.models.Model(inputs = inputs, outputs = output)
    model.compile("adam",loss="mse")
    return model


def learn_from_database(model, database, FLAGS):
    globals().update(FLAGS)
    indices_hand_pos = slice(0*n_steps,24*n_steps)
    indices_hand_vel = slice(24*n_steps,48*n_steps)
    indices_pen_pos = slice(48*n_steps,51*n_steps)
    indices_pen_rot = slice(51*n_steps,55*n_steps)
    indices_pen_vel = slice(55*n_steps,58*n_steps)
    indices_pen_rotvel = slice(58*n_steps,61*n_steps)
    goal_spaces_indices = [indices_hand_pos,indices_hand_vel,indices_pen_pos,indices_pen_rot,indices_pen_vel,indices_pen_rotvel]
    contexts, outcomes, actions = database[:,:context_dim], database[:,context_dim:-action_dim], database[:,-action_dim:]

    '''
    outcomes is a sequence of observations
    we prepend to it the context,
    and append a set of zeros
    and prepend a zeros action
    this is because from context, we can't predict an action yet
    and then we predict an action per outcome
    but the last section of the action vector is actually a goal parameter (part of the DMP parametrization), that we predict, without feeding an oucome

    we assume n_dmp_basis == n_steps
    '''
    assert n_dmp_basis == n_steps

    outcomes = outcomes.reshape((database.shape[0],context_dim,n_steps))
    outcomes = np.concatenate([outcomes,np.zeros(outcomes.shape[0:2]+(1,))],2)
    outcomes = np.concatenate([np.expand_dims(contexts,2),outcomes],2)
    outcomes = outcomes.transpose(0,2,1)

    masks = np.zeros((database.shape[0],context_dim*(n_steps+2)))
    masks[:,:context_dim] = 1
    for i in range(database.shape[0]):
        goal_space = np.random.randint(6)
        outcome_slice = goal_spaces_indices[goal_space]
        memory_slice = slice(context_dim+outcome_slice.start,context_dim+outcome_slice.stop)
        masks[i,memory_slice] = 1
    masks = masks.reshape(outcomes.shape)

    actions = actions.reshape((-1,n_actuators,(n_dmp_basis+1)))
    actions = actions.transpose(0,2,1)
    actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[2])),actions],1)

    '''
    both outcomes and actions have dimensions
    num_samples,time,dimension
    '''

    model.fit(outcomes*masks,actions,epochs=20, verbose=0)
    return model
