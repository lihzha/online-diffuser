from .registration import register_environments
from .env_wrapper import gym_env_wrapper, d4rl_env_wrapper

registered_environments = register_environments()