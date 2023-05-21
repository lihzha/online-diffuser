import gymnasium as gym
import panda_gym
import cv2
import numpy as np
env = gym.make("PandaPush-v3", render_mode="rgb_array")
observation, info = env.reset()
obs = []
ach = []
tar = []
for _ in range(30000):
    current_position = observation["observation"][0:3]
    desired_position = observation["desired_goal"][0:3]
    action = 5.0 * (desired_position - current_position)
    observation, reward, terminated, truncated, info = env.step(action)
    obs.append(observation['observation'])
    # if observation['achieved_goal'][0] > 0.16:
    #     print(observation['achieved_goal'])
    #     cv2.imwrite('reach.png',env.render())
    ach.append(observation['achieved_goal'])
    tar.append(observation['desired_goal'])
    # cv2.imwrite('reach.png',env.render())
    if terminated or truncated:
        observation, info = env.reset()
obs = np.array(obs)
ach = np.array(ach)
tar = np.array(tar)
np.save('obs.npy',obs)
np.save('achived.npy',ach)
np.save('desired.npy',tar)
env.close()