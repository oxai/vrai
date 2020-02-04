
#from latentgoalexplo.actors import exploactors
#
#import latentgoalexplo
#
#latentgoalexplo.environments

####
import pickle
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import gym
# import mujoco_py

env=gym.make("HandManipulatePen-v0")

env.reset()

env.observation_space["observation"]


context_dim = 61
n_simulation_steps = 30
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


#%%

outcome = []
for i in range(10):
    action = env.action_space.sample()
    results = env.step(action)
    obs = results[0]["observation"]
    outcome.append(obs)
    # env.render()

# outcome = np.stack(outcome)
####
env.action_space

env.env.sim.model.joint_name2id(env.env.sim.model.joint_names[2])

sim = env.env.sim
joints = env.env.sim.model.joint_names
n_joints = len(joints)

sim.data.ctrl


# joints

sim = env.env.sim

# sim.data.get_joint_qpos("object:joint").shape
#
for i in range(n_joints):
    print(sim.data.get_joint_qpos(env.env.sim.model.joint_names[i]))
    print(sim.data.get_joint_qvel(env.env.sim.model.joint_names[i]))


indices_hand_pos = slice(0*n_steps,24*n_steps)
indices_hand_vel = slice(24*n_steps,48*n_steps)
indices_pen_pos = slice(48*n_steps,51*n_steps)
indices_pen_rot = slice(51*n_steps,55*n_steps)
indices_pen_vel = slice(55*n_steps,58*n_steps)
indices_pen_rotvel = slice(58*n_steps,61*n_steps)

goal_spaces_indices = [indices_hand_pos,indices_hand_vel,indices_pen_pos,indices_pen_rot,indices_pen_vel,indices_pen_rotvel]
goal_spaces_names = ["hand_pos","hand_vel","pen_pos","pen_rot","pen_vel","pen_rotvel"]
n_goal_spaces = len(goal_spaces_indices)
# goal_outcomes = []
# for indices in goal_spaces_indices:
#     goal_outcomes.append(outcome[:,indices])

def goal_reward(goal, outcome, context):
    #TODO: I was lazy; really I should make it accept goal_space as argument
    return np.linalg.norm(goal - outcome)/np.sqrt(goal.shape[0])

###

# Instatiate metapolicies with random exploration


def goal_policy(goal_space, context):
    indices = goal_spaces_indices[goal_space]
    #TODO: check ranges are right
    goal_dim = (indices.stop-indices.start)
    return 2*np.random.rand(goal_dim)-1

intrinsic_rewards = np.array([1 for index in goal_spaces_indices])
def goal_space_probabilities(intrinsic_rewards):
    # probs = np.exp(intrinsic_rewards*(intrinsic_rewards>0)-np.Inf*(intrinsic_rewards<=0))
    # probs = np.exp(intrinsic_rewards/np.sum(intrinsic_rewards))*(intrinsic_rewards>0)
    # probs = np.exp(intrinsic_rewards/(np.sum(intrinsic_rewards)+0.01))*(intrinsic_rewards>0)
    probs = np.exp(intrinsic_rewards)*(intrinsic_rewards>0)
    if not np.any(probs>0):
        probs = np.ones(probs.shape)
    probs /= np.sum(probs)
    return probs

goal_space_probs = goal_space_probabilities(intrinsic_rewards)

def goal_space_policy(context):
    if np.random.rand() < 0.2:
        return np.random.choice(range(n_goal_spaces))
    else:
        return np.random.choice(range(n_goal_spaces), p=goal_space_probs)
    return None

# from scipy.spatial import KDTree
# KDTree()
# things = database[:,:61+610]
# indices = goal_spaces_indices[goal_space]
# goal_dim = (indices.stop-indices.start)*10
# things[:,:indices]/=np.sqrt(goal_dim)
default_mask = np.zeros(memory_dim)
default_mask[:-action_dim] = 1
def find_memory(context, goal_space, goal, action=None, mask=default_mask):
    outcome_slice = goal_spaces_indices[goal_space]
    return find_memory_by_slice(context, outcome_slice, goal, action=None, mask=default_mask)

def find_memory_by_slice(context, outcome_slice, goal, action=None, mask=default_mask):
    #TODO: This is a very hacky function, that is very specific to our implementation, rather than being general
    #it works because the goal and outcomes spaces are the same, but it's fine.
    query_vector = np.zeros(memory_dim)
    query_vector[:context_dim] = context
    #indices corresponding to the part of the outcome which we are querying against
    memory_slice = slice(context_dim+outcome_slice.start,context_dim+outcome_slice.stop)
    query_vector[memory_slice] = goal/np.sqrt(goal.shape[0])
    mask[memory_slice] /= goal/np.sqrt(goal.shape[0])
    if action is not None:
        query_vector[-action_dim:] = action
    distances = np.linalg.norm((database - query_vector)*mask, axis=1)
    index_of_memory = np.argmin(distances,axis=0)
    return index_of_memory

def meta_policy(goal_space, goal, context):
    if evaluating:
        outcome_slice = slice(goal_spaces_indices[2].start,goal_spaces_indices[3].stop)
        mask = np.zeros(memory_dim)
        memory_slice = slice(context_dim+outcome_slice.start,context_dim+outcome_slice.stop)
        mask[memory_slice] = 1
        index_of_memory = find_memory_by_slice(context, outcome_slice, goal, mask=mask)
    else:
        index_of_memory = find_memory(context, goal_space, goal)
    action = database[index_of_memory,-action_dim:]
    return action

def exploration_meta_policy(goal_space, goal, context):
    action = meta_policy(goal_space, goal, context)
    action += 0.1*np.random.randn(*action.shape)
    return np.clip(action, -1, 1)

# running_average_window_size = 5
running_average_weighting = 0.5
def update_intrinsic_reward(intrinsic_rewards, goal_space, goal, context, outcome):
    index_of_memory = find_memory(context, goal_space, goal)
    old_outcomes = database[index_of_memory,context_dim:-action_dim:]
    old_outcome_for_goal_space = old_outcomes[goal_spaces_indices[goal_space]]
    current_outcome_for_goal_space = outcome[goal_spaces_indices[goal_space]]
    # old_outcomes.shape
    learning_progress = goal_reward(goal,current_outcome_for_goal_space, context) - goal_reward(goal,old_outcome_for_goal_space, context)
    print(learning_progress)
    w = running_average_weighting
    r = intrinsic_rewards[goal_space]
    intrinsic_rewards[goal_space] = r*w + learning_progress*(1-w)
    return intrinsic_rewards

def secret_key():
    return "donkey balls"

def update_exploration_policy(context, outcome, action_parameter):
    global database
    database = np.concatenate([database, np.expand_dims(np.concatenate([context, outcome, action_parameter]),0)], axis=0)

def update_goal_space_policy():
    global goal_space_probs
    goal_space_probs = goal_space_probabilities(intrinsic_rewards)


from scipy.integrate import ode

def basis_function(t,t0):
    return np.exp(-0.5*(t-t0)**2/((n_simulation_steps/n_dmp_basis)**2))

def basis_functions(t,x,g,w,y0):
    phis = np.array(list(map(lambda t0: basis_function(t,t0), np.linspace(0,n_simulation_steps,n_dmp_basis)))).T
    return x*(g-y0)*np.matmul(w,phis)/np.sum(phis)

# variables = np.tile(np.array([0,0,1]),(n_actuators,1))
# w, g = action_parameter[:-n_actuators],action_parameter[-n_actuators:]
env.relative_control = True
def dmp(t, variables, w, g):
    #TODO: make it relative to current position, rather than reset. Need to set env.relative_control = True
    y0 = np.zeros(n_actuators)
    alphay = 0.3
    betay = 1.0
    alphax = 0.1
    variables = variables.reshape((n_actuators,3))
    y,v,x = variables[:,0],variables[:,1],variables[:,2]
    vdot = alphay*(betay*(g-y)-v) + basis_functions(t,x,g,w.reshape((n_actuators,n_dmp_basis)),y0)
    ydot = v
    xdot = -alphax*x
    return np.stack([ydot,vdot,xdot],axis=1).reshape((n_actuators*3))

#%%
# solver = ode(dmp)
# # dmp(0,np.array([0,1,1]),np.random.rand(10),0.5)
# solver.set_initial_value(np.tile(np.array([0,0,1]),(n_actuators,1)).reshape(-1),0).set_f_params(action_parameter[:-n_actuators],action_parameter[-n_actuators:])
# t1 = 10
# dt = 1
# ts = []
# ys = []
# while solver.successful() and solver.t < t1:
#     t = solver.t+dt
#     y = solver.integrate(solver.t+dt).reshape((n_actuators,3))[0,0]
#     ts.append(t)
#     ys.append(y)
#     # print(t, y)
#
# plt.plot(ts,ys)
#%%



def action_rollout(context,action_parameter,i):
    dt=1
    if i==0:
        solver = ode(dmp)
        solver.set_initial_value(np.tile(np.array([0,0,1]),(n_actuators,1)).reshape(-1),0)\
            .set_f_params(action_parameter[:-n_actuators],action_parameter[-n_actuators:])
        action_rollout.solver = solver
        return np.clip(action_rollout.solver.integrate(action_rollout.solver.t+dt).reshape((n_actuators,3))[:,0], -1,1)
    else:
        return np.clip(action_rollout.solver.integrate(action_rollout.solver.t+dt).reshape((n_actuators,3))[:,0], -1,1)
# solver.integrate(solver.t+dt)
# solver.t
#https://github.com/openai/gym/commit/78c416ef7bc829ce55b404b6604641ba0cf47d10.diff
#actions are just setting up the position of joints and actuators in hand (not sure why there are 20 and not 24 as in observations hmm)
# env.action_space
# env.relative_control

######################

# database.shape
# database = np.zeros((1,memory_dim)) #of (context,outcomes,actions)
import os
if os.path.exists("database.p"):
    database = pickle.load(open("database.p","rb"))
else:
    database = None
if os.path.exists("intrinsic_rewards.p"):
    intrinsic_rewards = pickle.load(open("intrinsic_rewards.p","rb"))
else:
    intrinsic_rewards = np.array([0.1 for index in goal_spaces_indices],dtype=np.float32)
goal_space_probs = goal_space_probabilities(intrinsic_rewards)

# action = env.action_space.sample()
results = env.reset()
context = results["observation"]
if evaluating:
    pen_goal = results["desired_goal"]
    goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
    goal = np.reshape(goal.T,(-1))

#some random exploration
if database is None: #cold start
    print("random warming up")
    reset_env = False
    memories = 0
    while memories < 1000:

        observations = []
        action_parameter = 2*np.random.rand(action_dim)-1
        for i in range(n_simulation_steps):
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
                if rendering:
                    env.render()
                observations.append(obs)

        if reset_env:
            reset_env = False
            continue
        else:
            memories+=1

        outcome = np.reshape(np.stack(observations).T, (outcome_dim))
        if database is None and memories == 1:
            database = np.expand_dims(np.concatenate([context, outcome, action_parameter]),0)
            # print(database.shape)
        else:
            update_exploration_policy(context, outcome, action_parameter)
else:
    memories = database.shape[0]

print("active goal babbling")
reset_env = False
for iteration in range(200000):
    print("iteration",iteration)
    goal_space = goal_space_policy(context)

    if not evaluating:
        goal = goal_policy(goal_space, context)

    #USE EXPLORATION POLICY
    if evaluating:
        action_parameter = meta_policy(goal_space, goal, context)
    else:
        action_parameter = exploration_meta_policy(goal_space, goal, context)
    # print(action_parameter)

    observations = []
    for i in range(n_simulation_steps):
        action = action_rollout(context,action_parameter, i)
        results = env.step(action)
        obs = results[0]["observation"]
        done = results[2]
        if done:
            print("reseting environment")
            results = env.reset()
            if evaluating:
                pen_goal = results["desired_goal"]
                goal = np.tile(np.expand_dims(pen_goal,0),(n_steps,1))
                goal = np.reshape(goal.T,(-1))
            reset_env = True
            break
        if i % outcome_sampling_frequency == 0:
            if rendering:
                env.render()
            observations.append(obs)

    if reset_env:
        reset_env = False
        continue
    else:
        memories += 1
    # np.stack(observations).shape
    # outcome = np.concatenate(observations)
    outcome = np.reshape(np.stack(observations).T, (outcome_dim))
    context = observations[-1]

    if not evaluating:
        intrinsic_rewards = update_intrinsic_reward(intrinsic_rewards, goal_space, goal, context, outcome)
        print(goal_spaces_names)
        print(intrinsic_rewards)

    #this updates knowledge too
    # database.shape
    # action_parameter.shape

    # np.expand_dims(np.concatenate([context, outcome, action_parameter]),0).shape

    if not evaluating:
        update_exploration_policy(context, outcome, action_parameter)
        update_goal_space_policy()
        if iteration % save_freq:
            pickle.dump(database, open("database.p","wb"))
            pickle.dump(intrinsic_rewards, open("intrinsic_rewards.p","wb"))

# pickle.dump(database, open("database.p","wb"))
# pickle.dump(intrinsic_rewards, open("intrinsic_rewards.p","wb"))
