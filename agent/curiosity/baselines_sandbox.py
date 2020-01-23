import gym
import os

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import DDPG
import numpy as np
import tensorflow as tf

from stable_baselines.ddpg.policies import FeedForwardPolicy
# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           # net_arch=[dict(pi=[256,256,256], vf=[256,256,256])],
                                           layers=[256]*3,
                                           act_fun=tf.nn.relu,
                                           layer_norm=False,
                                           feature_extraction="mlp")


#%%
env_name="FetchReach-v1"

keys = ['observation', 'desired_goal']
try: # for modern Gym (>=0.15.4)
    from gym.wrappers import FilterObservation, FlattenObservation
    env_fun = lambda : FlattenObservation(FilterObservation(gym.make(env_name),keys))
except ImportError: # for older gym (<=0.15.3)
    from gym.wrappers import FlattenDictWrapper
    env_fun = lambda : FlattenDictWrapper(gym.make(env_name),keys)
# env = DummyVecEnv([lambda: gym.make("Reacher-v2")])
env = DummyVecEnv([env_fun])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=200.)
# env = gym.make("FetchReach-v1")
#%%

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))


model = DDPG(CustomPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, critic_l2_reg=1.0, batch_size=256, buffer_size=1000000,\
            random_exploration=0.3, observation_range=(-200,200),actor_lr=1e-3, critic_lr=1e-3,tensorboard_log="summaries")
            # random_exploration=0.3, normalize_observations=True, observation_range=(-200,200),actor_lr=1e-3, critic_lr=1e-3)
model.learn(total_timesteps=2e5)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/tmp/"
model.save(log_dir + "ppo_reacher")
env.save(os.path.join(log_dir, "vec_normalize.pkl"))


# # Custom MLP policy of two layers of size 16 each
# class CustomDDPGPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
#                                            layers=[16, 16],
#                                            layer_norm=False,
#                                            feature_extraction="mlp")


# model = DDPG(CustomPolicy, 'Pendulum-v0', verbose=1)
# model = DDPG(CustomPolicy, 'FetchReach-v1', verbose=1, param_noise=param_noise, action_noise=action_noise, critic_l2_reg=1.0, batch_size=256, buffer_size=1000000,\
#                 random_exploration=0.3)
