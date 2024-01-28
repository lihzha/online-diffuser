from diffuser.datasets.d4rl import load_environment
from diffuser.utils.rendering import Maze2dRenderer
from stable_baselines3 import PPO
import numpy as np

env = load_environment("maze2d-large-v1")
renderer = Maze2dRenderer("maze2d-large-v1")
model = PPO("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
score_mean = []
total_reward_mean = []
for t in range(1000):
    model.learn(total_timesteps=100_000)

    score_list = []
    total_reward_list = []
    eval_epoch = 20
    for j in range(eval_epoch):
        total_reward = 0
        obs = vec_env.reset()
        rollout = [obs.copy().squeeze()]
        for i in range(env.max_episode_steps):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward
            rollout.append(obs.copy().squeeze())
        print(total_reward)
        score = env.get_normalized_score(total_reward)
        score_list.append(score)
        total_reward_list.append(total_reward)
        renderer.composite(f'ppo_evaluation_{t}_{j}.png', np.array(rollout)[None], ncol=1)
    score_array = np.array(score_list)
    total_reward_array = np.array(total_reward_list)
    print('score mean and std:', score_array.mean(), score_array.std()/np.sqrt(eval_epoch))
    print('reward mean and std:', total_reward_array.mean(), total_reward_array.std()/np.sqrt(eval_epoch))
    score_mean.append(score_array.mean())
    total_reward_mean.append(total_reward_array.mean())
print(score_mean)
print(total_reward_mean)
np.save('score.npy', np.array(score_mean))
np.save('total_reward.npy', np.array(total_reward_mean))
print(np.array(score_mean).mean())
print(np.array(total_reward_mean).mean())
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()