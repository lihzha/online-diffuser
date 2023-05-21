import gymnasium as gym
import panda_gym
def env_wrapper(env, render_mode):
    env = gym.make(env, render_mode=render_mode)
    return env