'''
Using a neural network for metapolicy, in parallel to fast memory-based metapolicy
'''

import pickle
from absl import app
import matplotlib.pyplot as plt
import os,sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
AGENT_DIR = os.path.join(ROOT_DIR, 'agent')
MEMORIES_DIR = os.path.join(THIS_DIR,"memories")
# ENVIRONMENT_DIR = os.path.join(ROOT_DIR, 'environment')
# EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
sys.path.append(ROOT_DIR)
sys.path.append(AGENT_DIR)

import sys
# %matplotlib inline
import numpy as np
import gym
# import mujoco_py

env=gym.make("HandManipulatePen-v0")
env.reset()
env.observation_space["observation"]
env.relative_control = True
from robot_hand_utils import get_actuator_center

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_dmp_basis",20,"number of basis functions used in the dynamical movement primitive")
flags.DEFINE_integer("n_simulation_steps",80,"number of simulation steps of each individual action rollout")
flags.DEFINE_integer("outcome_sampling_frequency",4,"frequency of observations which are saved into the outcome vector")
flags.DEFINE_bool("rendering",False,"whether to render the environment or not")
flags.DEFINE_bool("evaluating",False,"whether we are evaluating on the target goals, or still learning/exploring")
flags.DEFINE_integer("save_freq",1000,"the frequency by which to save memories")


def goal_reward(goal_space, goal, outcome, context, precision=1.0):
    # return np.tanh(-precision*(np.linalg.norm(goal - outcome))**2/goal.shape[0])
    a = 0.5
    x = (np.linalg.norm(goal - outcome))**2/goal.shape[0]
    return -np.tanh(a*precision*x-1/precision)

def goal_policy(goal_space, context):
    #goal is of size len(goal_spaces_indices[goal_space])
    indices = goal_spaces_indices[goal_space]
    goal_dim = (indices.stop-indices.start)
    #observation space has limits between -inf and inf (see env.observation_space.high and low)
    return np.random.randn(goal_dim)

def goal_space_probabilities(intrinsic_rewards):
    #TODO: check if there are better bandit algorithms
    # probs = np.exp(intrinsic_rewards*(intrinsic_rewards>0)-np.Inf*(intrinsic_rewards<=0))
    # probs = np.exp(intrinsic_rewards/np.sum(intrinsic_rewards))*(intrinsic_rewards>0)
    # probs = np.exp(intrinsic_rewards/(np.sum(intrinsic_rewards)+0.01))*(intrinsic_rewards>0)
    probs = np.exp(intrinsic_rewards)*(intrinsic_rewards>0)
    if not np.any(probs>0):
        probs = np.ones(probs.shape)
    probs /= np.sum(probs)
    return probs

def goal_space_policy(context):
    if np.random.rand() < 0.2:
        return np.random.choice(range(n_goal_spaces))
    else:
        return np.random.choice(range(n_goal_spaces), p=goal_space_probs)
    return None

# The function that takes a goal and returns an action_parameter to perform
def meta_policy_inner(goal_space, goal, context, goodness_threshold=-1):
    # if evaluating then the goal is special, it's the both the pen and position and rotation
    used_neural_net = False
    if evaluating:
        ###different types of evaluation###
        indices_pen_pos = slice(48*n_steps,51*n_steps)
        indices_pen_rot = slice(51*n_steps,55*n_steps)
        outcome_slice = slice(indices_pen_pos.start,indices_pen_rot.stop)
        # outcome_slice = slice(goal_spaces_indices[2].start,goal_spaces_indices[2].stop)
        # outcome_slice_one_time = slice(goal_spaces_indices[2].start//n_steps,goal_spaces_indices[2].stop//n_steps)
        # outcome_slice = slice(goal_spaces_indices[3].start,goal_spaces_indices[3].stop)
        # outcome_slice_one_time = slice(goal_spaces_indices[3].start//n_steps,goal_spaces_indices[3].stop//n_steps)
        mask = np.zeros(memory_dim)
        memory_slice = slice(context_dim+outcome_slice.start,context_dim+outcome_slice.stop)
        mask[memory_slice] = 1
        mask[:context_dim] = 1
        index_of_memory, goodness = find_memory_by_slice(context, outcome_slice, goal, mask=mask)
        print("goodness", goodness)
        if goodness > 5:
            outcome_slice = slice(indices_pen_pos.start,indices_pen_rot.stop)
            # outcome_slice = slice(goal_spaces_indices[3].start,goal_spaces_indices[3].stop)
            memory_slice = slice(outcome_slice.start,outcome_slice.stop)
            outcomes = np.zeros(outcome_dim)
            outcomes[memory_slice] = goal
            outcomes = np.expand_dims(outcomes,0)
            outcomes = outcomes.reshape((1,context_dim,n_steps))
            outcomes = np.concatenate([outcomes,np.zeros(outcomes.shape[0:2]+(1,))],2)
            context = np.expand_dims(context,0)
            outcomes = np.concatenate([np.expand_dims(context,2),outcomes],2)
            outcomes = outcomes.transpose(0,2,1)
            action = meta_policy_nn_model(outcomes.astype("float32")).numpy()[0,1:,:].transpose(1,0).reshape((-1,))
            print(action.shape)
            used_neural_net = True
        else:
            action = database[index_of_memory,-action_dim:]
    else:
        index_of_memory, goodness = find_memory_by_reward(context, goal_space, goal)
        print("goodness",goodness)
        p = 0.05
        # with probability p or if goodness of the memory-based learner action is good, then use that
        # otherwise use the neural net
        if not np.random.rand() < p and goodness < goodness_threshold:
            outcome_slice = goal_spaces_indices[goal_space]
            memory_slice = slice(outcome_slice.start,outcome_slice.stop)
            outcomes = np.zeros(outcome_dim)
            outcomes[memory_slice] = goal
            outcomes = np.expand_dims(outcomes,0)
            outcomes = outcomes.reshape((1,context_dim,n_steps))
            outcomes = np.concatenate([outcomes,np.zeros(outcomes.shape[0:2]+(1,))],2)
            context = np.expand_dims(context,0)
            outcomes = np.concatenate([np.expand_dims(context,2),outcomes],2)
            outcomes = outcomes.transpose(0,2,1)
            # meta_policy_nn_model.get_weights()
            old_weights = meta_policy_nn_model.get_weights()
            perturbed_weights = perturb_weights(old_weights)
            meta_policy_nn_model.set_weights(perturbed_weights)
            action = meta_policy_nn_model(outcomes.astype("float32")).numpy()[0,1:,:].transpose(1,0).reshape((-1,))
            meta_policy_nn_model.set_weights(old_weights)
            # print(action.shape)
            used_neural_net = True
        else:
            action = database[index_of_memory,-action_dim:]

    action[:-n_actuators] = np.clip(action[:-n_actuators], -1, 1)
    actuator_center = get_actuator_center(env)
    action[-n_actuators:] = np.clip(action[-n_actuators:], -1 - actuator_center, 1 - actuator_center)
    return action, goodness, used_neural_net

# this is to perform parameter noise (as opposed to adding noise to actions)
def perturb_weights(old_weights):
    perturbed_weights = []
    for weight in old_weights:
        perturbed_weights.append(weight+0.1*np.random.randn(*weight.shape))
    return perturbed_weights

#wrapper when goodness needs not be returned
def meta_policy(goal_space,goal,context):
    return meta_policy_inner(goal_space, goal, context)[0]

def exploration_meta_policy(goal_space, goal, context):
    # goodness_threshold = -1
    goodness_threshold = -1.5
    action, goodness, used_neural_net = meta_policy_inner(goal_space, goal, context, goodness_threshold)
    if used_neural_net: #you used memory-based learning, so add noise to actions (as opposed to to parameters when using the neural net)
        action += 0.1*np.random.randn(*action.shape)
    #clipping so that the added noise doesn't exceed the limits of the actuators
    action[:-n_actuators] = np.clip(action[:-n_actuators], -1, 1)
    actuator_center = get_actuator_center(env)
    action[-n_actuators:] = np.clip(action[-n_actuators:], -1 - actuator_center, 1 - actuator_center)
    return action

# running_average_window_size = 5
running_average_weighting = 0.9
def update_intrinsic_reward(intrinsic_rewards, goal_space, goal, context, outcome):
    index_of_memory,_ = find_memory_by_reward(context, goal_space, goal)
    old_outcomes = database[index_of_memory,context_dim:-action_dim:]
    old_outcome_for_goal_space = np.expand_dims(old_outcomes,0)
    current_outcome_for_goal_space = np.expand_dims(outcome,0)
    # old_outcomes.shape
    current_reward = reward_funs(goal_space)(context,current_outcome_for_goal_space,goal)
    previous_reward = reward_funs(goal_space)(context,old_outcome_for_goal_space,goal)
    # learning_progress = current_reward - previous_reward
    learning_progress = np.abs(current_reward - previous_reward)
    print("current_reward",current_reward)
    print("previous_reward",previous_reward)
    print("absolute learning_progress",learning_progress)
    w = running_average_weighting
    r = intrinsic_rewards[goal_space]
    intrinsic_rewards[goal_space] = r*w + learning_progress*(1-w)
    return intrinsic_rewards

def update_exploration_policy(context, outcome, action_parameter):
    global database
    database = np.concatenate([database, np.expand_dims(np.concatenate([context, outcome, action_parameter]),0)], axis=0)

def update_goal_space_policy():
    global goal_space_probs
    goal_space_probs = goal_space_probabilities(intrinsic_rewards)


def main(argv):

    '''PREPPING UP variables'''

    global FLAGS
    FLAGS = FLAGS.flag_values_dict()
    # print(FLAGS)
    globals().update(FLAGS)

    context_dim = env.observation_space["observation"].shape[0]
    FLAGS["context_dim"] = context_dim
    n_steps = n_simulation_steps//outcome_sampling_frequency
    FLAGS["n_steps"] = n_steps
    n_actuators = env.action_space.shape[0]
    FLAGS["n_actuators"] = n_actuators
    action_dim = n_actuators*(n_dmp_basis+1) # +1 for target position, in dmp parametrization
    FLAGS["action_dim"] = action_dim
    outcome_dim = context_dim*n_steps
    FLAGS["outcome_dim"] = outcome_dim
    FLAGS["memory_dim"] = context_dim+outcome_dim+action_dim
    globals().update(FLAGS)

    indices_hand_pos = slice(0*n_steps,24*n_steps)
    indices_hand_vel = slice(24*n_steps,48*n_steps)
    indices_pen_pos = slice(48*n_steps,51*n_steps)
    indices_pen_rot = slice(51*n_steps,55*n_steps)
    indices_pen_vel = slice(55*n_steps,58*n_steps)
    indices_pen_rotvel = slice(58*n_steps,61*n_steps)

    global n_goal_spaces, goal_spaces_indices, goal_spaces_names, reward_funs, find_memory_by_reward, find_memory_by_slice
    # goal_spaces_indices = [indices_hand_pos,indices_hand_vel,indices_pen_pos,indices_pen_rot,indices_pen_vel,indices_pen_rotvel]
    goal_spaces_indices = [indices_pen_pos,indices_pen_rot,indices_pen_vel,indices_pen_rotvel]
    # goal_spaces_names = ["hand_pos","hand_vel","pen_pos","pen_rot","pen_vel","pen_rotvel"]
    # goal_spaces_names = ["pen_pos","pen_rot","pen_vel","pen_rotvel"]
    goal_spaces_names = ["pen_pos","pen_rot"]
    goal_precisions = [0.7, 1.0, 2.0, 3.0, 10.0]
    goal_spaces_indices = [goal_spaces_indices[i] for i,g in enumerate(goal_spaces_names) for p in goal_precisions]
    goal_spaces_names = [goal_space_name+"_"+"{0:1}".format(precision) for goal_space_name in goal_spaces_names for precision in goal_precisions]
    n_goal_spaces = len(goal_spaces_indices)

    env.relative_control = True
    from dmp import DMP
    dmp = DMP(n_dmp_basis,n_simulation_steps,n_actuators)
    action_rollout = dmp.action_rollout


    #outcome is a flattned (row-major) version of an array of dimensions (observation_dim,n_steps)

    # goal_reward_vectorized = np.vectorize(goal_reward)
    def reward_funs(goal_space):
        precision = float(goal_spaces_names[goal_space].split("_")[-1])
        def reward_fun(context, outcomes, goal):
            outcome_slice = slice(goal_spaces_indices[goal_space].start,goal_spaces_indices[goal_space].stop)
            outcomes = outcomes[:,outcome_slice]
            return goal_reward(goal_space, goal, outcomes, context, precision)
        return reward_fun

    def find_memory_by_reward(context, goal_space, goal):
        outcomes = database[:,context_dim:-action_dim]
        rewards = reward_funs(goal_space)(context,outcomes,goal)
        # print("AAA", rewards.shape)
        goodness = -np.linalg.norm(database[:,:context_dim]-context,axis=1) + rewards
        index_of_memory = np.argmax(goodness,axis=0)
        return index_of_memory, goodness[index_of_memory]

    default_mask = np.zeros(memory_dim)
    default_mask[:context_dim] = 1
    def find_memory_by_slice(context, outcome_slice, goal, action=None, mask=default_mask):
        #TODO: This is a very hacky function, that is very specific to our implementation, rather than being general
        #it works because the goal and outcomes spaces are the same, but it's fine.
        query_vector = np.zeros(memory_dim)
        query_vector[:context_dim] = context
        #indices corresponding to the part of the outcome which we are querying against
        memory_slice = slice(context_dim+outcome_slice.start,context_dim+outcome_slice.stop)
        query_vector[memory_slice] = goal/np.sqrt(goal.shape[0])
        mask[memory_slice] = 1
        if action is not None:
            query_vector[-action_dim:] = action
        distances = np.linalg.norm((database - query_vector)*mask, axis=1)
        index_of_memory = np.argmin(distances,axis=0)
        return index_of_memory, distances[index_of_memory]


    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if evaluating: assert size == 1
    if not evaluating: assert size == 2

    #initialize NN
    from meta_policy_neural_net import make_meta_policy_nn_model

    global meta_policy_nn_model
    meta_policy_nn_model = make_meta_policy_nn_model(FLAGS)

    if os.path.exists(os.path.join(THIS_DIR,"model_weights.p")):
        updated_model_weigths = pickle.load(open(os.path.join(THIS_DIR,"model_weights.p"),"rb"))
        meta_policy_nn_model.set_weights(updated_model_weigths)

    if rank == 0:
        # yeah this is uggly, okkk
        global database
        global intrinsic_rewards
        global goal_space_probs
        files = os.listdir(MEMORIES_DIR)
        # print(list(files))
        files = [os.path.join(MEMORIES_DIR, f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x),reverse=True)
        MAX_MEMORY_SIZE = 10000
        # MAX_MEMORY_SIZE = 100000
        print(files)
        if len(files)>0:
            for ii,filename in enumerate(files):
                if ii==0:
                    database = np.load(filename)
                else:
                    database = np.concatenate([database,np.load(filename)],0)
                if len(database) >MAX_MEMORY_SIZE:
                    break
        else:
            database = None
        if os.path.exists(os.path.join(THIS_DIR,"intrinsic_rewards.p")):
            intrinsic_rewards = pickle.load(open(os.path.join(THIS_DIR,"intrinsic_rewards.p"),"rb"))
        else:
            intrinsic_rewards = np.array([0.01 for index in goal_spaces_indices],dtype=np.float32)
        goal_space_probs = goal_space_probabilities(intrinsic_rewards)

        '''INITIALIZE ENVIRONMENT'''

        # action = env.action_space.sample()
        results = env.reset()
        context = results["observation"]
        if evaluating:
            pen_goal = results["desired_goal"]
            # pen_goal = pen_goal[:3]
            # pen_goal = pen_goal[3:]
            goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
            goal = np.reshape(goal.T,(-1))

        '''INITILIAZE MEMORY DATABASE VIA RANDOM EXPLORATION'''

        #some random exploration
        if database is None: #cold start
            print("random warming up")
            reset_env = False
            memories = 0
            while memories < 1000:
                observations = []
                action_parameter = 2*np.random.rand(action_dim)-1
                for i in range(n_simulation_steps):
                    # print(i)
                    if rendering:
                        env.render()
                    action = action_rollout(context, action_parameter, i)
                    results = env.step(action)
                    obs = results[0]["observation"]
                    done = results[2]
                    if done:
                        print("reseting environment")
                        results = env.reset()
                        # obs = results["observation"]
                        reset_env = True
                        break
                    if i % outcome_sampling_frequency == 0:
                        observations.append(obs)

                if reset_env:
                    reset_env = False
                    continue
                else:
                    print("Adding memory")
                    sys.stdout.flush()
                    memories+=1

                outcome = np.reshape(np.stack(observations).T, (outcome_dim))
                if database is None and memories == 1:
                    database = np.expand_dims(np.concatenate([context, outcome, action_parameter]),0)
                    # print(database.shape)
                else:
                    update_exploration_policy(context, outcome, action_parameter)
        else:
            memories = database.shape[0]
            if not evaluating: comm.send(True, dest=1, tag=11) #signal to send to consolidation code to continue going on

        '''TRAINING LOOP'''
        print("active goal babbling")
        reset_env = False
        for iteration in range(200000):
            print("iteration",iteration)
            # this chooses one of the goal_spaces as an index from 0 to len(goal_spaces_indices)-1
            goal_space = goal_space_policy(context)

            if not evaluating:
                if comm.Iprobe(source=1, tag=12):
                    updated_model_weigths = comm.recv(source=1, tag=12)
                    meta_policy_nn_model.set_weights(updated_model_weigths)
                #goal is of size len(goal_spaces_indices[goal_space])
                goal = goal_policy(goal_space, context)

            if evaluating:
                action_parameter = meta_policy(goal_space, goal, context)
            else:
                #USE EXPLORATION POLICY
                action_parameter = exploration_meta_policy(goal_space, goal, context)

            observations = []
            for i in range(n_simulation_steps):
                # print(context.shape)
                action = action_rollout(context,action_parameter, i)
                results = env.step(action)
                if rendering:
                    env.render()
                obs = results[0]["observation"]
                # print(obs)
                done = results[2]
                if done:
                    print("reseting environment")
                    results = env.reset()
                    if evaluating:
                        pen_goal = results["desired_goal"]
                        # pen_goal = pen_goal[:3]
                        # pen_goal = pen_goal[3:]
                        goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
                        goal = np.reshape(goal.T,(-1))
                    reset_env = True
                    break
                if i % outcome_sampling_frequency == 0:
                    observations.append(obs)

            if reset_env:
                reset_env = False
                # continue
            else:
                memories += 1
                outcome = np.reshape(np.stack(observations).T, (outcome_dim))

                if not evaluating:
                    intrinsic_rewards = update_intrinsic_reward(intrinsic_rewards, goal_space, goal, context, outcome)
                    print(goal_spaces_names)
                    print(intrinsic_rewards)
                    sys.stdout.flush()

                if not evaluating:
                    update_exploration_policy(context, outcome, action_parameter)
                    update_goal_space_policy()
            if not evaluating:
                if iteration % save_freq == save_freq - 1:
                    print("Saving new batch of memories")
                    sys.stdout.flush()
                    # pickle.dump(database, open("database.p","wb"))
                    database = database[-MAX_MEMORY_SIZE:]
                    np.save(os.path.join(MEMORIES_DIR,"database_"+str(iteration)+".npy"),database[-save_freq:])
                    pickle.dump(intrinsic_rewards, open(os.path.join(THIS_DIR,"intrinsic_rewards.p"),"wb"))
                    comm.send(True, dest=1, tag=11) #signal to send to consolidation code to continue going on
            context = observations[-1]

        comm.send(False, dest=1, tag=11)
    if rank == 1:
        from meta_policy_neural_net import learn_from_database
        while comm.recv(source=0, tag=11):
        # while True:
            # files = filter(os.path.isfile, os.listdir("memories"))
            files = os.listdir(MEMORIES_DIR)
            # print(list(files))
            files = [os.path.join(MEMORIES_DIR, f) for f in files] # add path to each file
            files.sort(key=lambda x: os.path.getmtime(x),reverse=True)
            # print(files)
            sys.stdout.flush()
            for filename in files:
                # if True:
                if not comm.Iprobe(source=0, tag=11):
                    database = np.load(filename)
                    print("Training on",filename)
                    sys.stdout.flush()
                    meta_policy_nn_model = learn_from_database(meta_policy_nn_model,database,FLAGS)
                    # sys.stdout.flush()
                    updated_model_weigths = meta_policy_nn_model.get_weights()
                    comm.send(updated_model_weigths, dest=0, tag=12)
                    pickle.dump(updated_model_weigths, open(os.path.join(THIS_DIR,"model_weights.p"),"wb"))
                else:
                    break


if __name__ == '__main__':
  app.run(main)
