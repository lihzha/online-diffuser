import torch
import logging
import numpy as np
import gymnasium as gym
from buffer import ReplayBuffer, PrioritizedReplayBuffer, get_buffer
from copy import deepcopy
from dotmap import DotMap
from omegaconf import OmegaConf
from hydra.utils import instantiate
from utils import merge_videos, visualize, set_seed_everywhere, get_space_shape
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval(env, agent, episodes, seed):
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=episode + seed)
        done, truncated = False, False

        while not (done or truncated):
            state, _, done, truncated, info = env.step(agent.get_action(state))
        returns.append(info['episode']['r'].item())
    return np.mean(returns), np.std(returns)


def train(cfg, seed, log_dict, idx, logger, barrier):
    env = RecordEpisodeStatistics(gym.make(cfg.env_name, render_mode="rgb_array"))
    set_seed_everywhere(env, seed)
    state_size = get_space_shape(env.observation_space)
    action_size = get_space_shape(env.action_space)
    buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed)
    agent = instantiate(cfg.agent, state_size=state_size, action_size=action_size, action_space=env.action_space, device=device)

    # get_attr of omega_conf is slow, so we convert it to dotmap
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

    logger.info(f"Training seed {seed} for {cfg.timesteps} timesteps with {agent} and {buffer}")
    eval_env = deepcopy(env)
    local_log_dict = {key: [] for key in log_dict.keys()}

    done, truncated, best_reward = False, False, -np.inf
    state, _ = env.reset(seed=seed)
    for step in range(1, cfg.timesteps + 1):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False
            local_log_dict['train_returns'].append(info['episode']['r'].item())
            local_log_dict['train_steps'].append(step - 1)

        action = agent.get_action(state, sample=True)

        next_state, reward, done, truncated, info = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, weights=weights)
                buffer.update_priorities(tree_idxs, ret_dict['td_error'])

            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch)
            else:
                raise RuntimeError("Unknown buffer")

            for key in ret_dict.keys():
                local_log_dict[key].append(ret_dict[key])

        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            local_log_dict['eval_steps'].append(step - 1)
            local_log_dict['eval_returns'].append(eval_mean)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward} and step {step}')
                agent.save(f'best_model_seed_{seed}.pt')

        if step % cfg.plot_interval == 0:
            for key in local_log_dict.keys():
                log_dict[key][idx] = local_log_dict[key]
            barrier.wait()
            if idx == 0:
                # prevent other processes from modifying the log_dict during visualization
                visualize(step, f'{agent} with {buffer}', log_dict)

    agent.save(f'final_model_seed_{seed}.pt')
    for key in local_log_dict.keys():
        log_dict[key][idx] = local_log_dict[key]

    # make sure all processes finish training before the final visualization
    barrier.wait()
    if idx == 0:
        visualize(step, f'{agent} with {buffer}', log_dict)
    env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x %
                      3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    agent.load(f'best_model_seed_{seed}.pt')  # use best model for visualization
    env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x %
                      3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
    merge_videos(f'final_videos_seed_{seed}')
    merge_videos(f'best_videos_seed_{seed}')
    env.close()
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
    return eval_mean
