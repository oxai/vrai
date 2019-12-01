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

def render_with_target(env, target):
    import mujoco_py
    env.sim.data.set_joint_qpos('target:joint', target)
    env.sim.data.set_joint_qvel('target:joint', np.zeros(6))
    if 'object_hidden' in env.sim.model.geom_names:
        hidden_id = env.sim.model.geom_name2id('object_hidden')
        env.sim.model.geom_rgba[hidden_id, 3] = 1.
    env.sim.forward()
    env.viewer = mujoco_py.MjViewer(env.sim)
    body_id = env.sim.model.body_name2id('robot0:palm')
    lookat = env.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        env.viewer.cam.lookat[idx] = value
    env.viewer.cam.distance = 0.5
    env.viewer.cam.azimuth = 55.
    env.viewer.cam.elevation = -25.
    env.viewer.render()
