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

import mujoco_py
def render_with_target(env, target):
    sim = env.sim
    sites_offset = (sim.data.site_xpos - sim.model.site_pos).copy()
    site_id = sim.model.site_name2id('target0')
    sim.model.site_pos[site_id] = target - sites_offset[0]
    sim.forward()
    body_id = env.sim.model.body_name2id('robot0:gripper_link')
    #lookat = env.sim.data.body_xpos[body_id]
    #for idx, value in enumerate(lookat):
    #    env.viewer.cam.lookat[idx] = value
    env.viewer.cam.distance = 2.5
    env.viewer.cam.azimuth = 132.
    env.viewer.cam.elevation = -14.
    env.viewer.render()

def setup_render(env):
    env.viewer = mujoco_py.MjViewer(env.sim)
    body_id = env.sim.model.body_name2id('robot0:gripper_link')
    lookat = env.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        env.viewer.cam.lookat[idx] = value
    env.viewer.cam.distance = 2.5
    env.viewer.cam.azimuth = 132.
    env.viewer.cam.elevation = -14.
    env.viewer.render()

