from diffuser.datasets.d4rl import load_environment

def gym_env_wrapper(env, render_mode):
    import gymnasium as gym
    import panda_gym
    env = gym.make(env, render_mode=render_mode)
    return env

def d4rl_env_wrapper(env):
    import gym
    env = load_environment(env)
    return env