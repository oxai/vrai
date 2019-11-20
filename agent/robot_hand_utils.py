import numpy as np

def get_actuator_center(env):
    sim = env.env.sim
    action = env.action_space.sample()
    ctrlrange = sim.model.actuator_ctrlrange
    actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
    actuation_center = np.zeros_like(action)
    for i in range(sim.data.ctrl.shape[0]):
        actuation_center[i] = sim.data.get_joint_qpos(
        sim.model.actuator_names[i].replace(':A_', ':'))
        for joint_name in ['FF', 'MF', 'RF', 'LF']:
            act_idx = sim.model.actuator_name2id(
            'robot0:A_{}J1'.format(joint_name))
            actuation_center[act_idx] += sim.data.get_joint_qpos(
            'robot0:{}J0'.format(joint_name))
    return actuation_center
