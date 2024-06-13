from stable_baselines3 import PPO
from pettingzoo.atari import wizard_of_wor_v3
import supersuit as ss
env = wizard_of_wor_v3.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
model = PPO('CnnPolicy', env, verbose=3, n_steps=16)
model.learn(total_timesteps=2000000)