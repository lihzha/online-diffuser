import os
import glob
import torch
import sys
import shutil
import random
import logging
import numpy as np
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from moviepy.editor import VideoFileClip, concatenate_videoclips


def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])


def get_log_dict(agent_name, manager=None, num_seeds=0):
    log_keys = ['critic_loss', 'actor_loss', 'td_error', 'eval_steps', 'eval_returns', 'train_steps', 'train_returns']
    if 'td3' in agent_name:
        log_keys.append('critic_loss_2')
    if 'sac' in agent_name:
        log_keys.append('alpha')
        log_keys.append('critic_loss_2')
        # log_keys.append('log_prob')
        # log_keys.append('log_prob1')
        # log_keys.append('log_prob2')
    if manager is None:
        return {key: [] for key in log_keys}
    else:
        return manager.dict({key: manager.list([[]] * num_seeds) for key in log_keys})


def visualize(step, title, log_dict):
    train_window, loss_window = 10, 200
    plt.figure(figsize=(20, 6))

    # # plot train and eval returns
    # plt.subplot(2,2,1)
    # plt.title('frame %s. score: %s' % (step, np.mean(log_dict['train_returns'][-1][-train_window:])))
    # if min([len(log_dict['train_steps'][i]) for i in range(len(log_dict['train_steps']))]) > train_window - 1:
    #     plot_scores(log_dict['train_returns'], log_dict['train_steps'], train_window, label='train')
    # if min([len(log_dict['eval_steps'][i]) for i in range(len(log_dict['eval_steps']))]) > 0:
    #     plot_scores(log_dict['eval_returns'], log_dict['eval_steps'], window=1, label='eval', color='C1')
    # plt.legend()
    # plt.ylabel('scores')
    # plt.xlabel('step')

    # # plot td losses
    # plt.subplot(2,2,2)
    # plt.title('critic metrics')
    # if 'critic_loss_2' in log_dict.keys():
    #     plot_scores(log_dict['critic_loss_2'], window=loss_window, label='critic_loss_2', color='C1')
    # plot_scores(log_dict['critic_loss'], window=loss_window, label='critic_loss', color='C0')
    # plt.xlabel('step')
    # plt.ylabel('critic loss')
    # plt.legend()
    # plt.subplot(2,2,3)

    # # plot actor metrics
    # lines = []
    # plt.title('actor metrics')
    # lines.append(plot_scores(log_dict['actor_loss'], window=loss_window, label='actor_loss'))
    # plt.xlabel('step')
    # plt.ylabel('actor_loss')
    # if 'alpha' in log_dict.keys():
    #     plt.twinx()  # instantiate a second axes that shares the same x-axis
    #     lines.append(plot_scores(log_dict['alpha'], window=loss_window, label='alpha', color='C1'))
    # plt.legend(lines, [line.get_label() for line in lines])

    # lines = []
    # plt.subplot(2, 2, 4)
    # plt.title('log_prob')
    # lines.append(plot_scores(log_dict['log_prob'], window=loss_window, label='log_prob', color='C0'))
    # plt.xlabel('step')
    # plt.ylabel('log prob')
    # if 'log_prob1' in log_dict.keys():
    #     lines.append(plot_scores(log_dict['log_prob1'], window=loss_window, label='mean', color='C1'))
    # if 'log_prob2' in log_dict.keys():
    #     lines.append(plot_scores(log_dict['log_prob2'], window=loss_window, label='std', color='C1'))
    # plt.legend(lines, [line.get_label() for line in lines])

    # plt.suptitle(title, fontsize=16)
    # plt.savefig('results.png')
    # plt.close()


    plt.subplot(1,3,1)
    plt.title('frame %s. score: %s' % (step, np.mean(log_dict['train_returns'][-1][-train_window:])))
    if min([len(log_dict['train_steps'][i]) for i in range(len(log_dict['train_steps']))]) > train_window - 1:
        plot_scores(log_dict['train_returns'], log_dict['train_steps'], train_window, label='train')
    if min([len(log_dict['eval_steps'][i]) for i in range(len(log_dict['eval_steps']))]) > 0:
        plot_scores(log_dict['eval_returns'], log_dict['eval_steps'], window=1, label='eval', color='C1')
    plt.legend()
    plt.ylabel('scores')
    plt.xlabel('step')

    # plot td losses
    plt.subplot(1,3,2)
    plt.title('critic metrics')
    if 'critic_loss_2' in log_dict.keys():
        plot_scores(log_dict['critic_loss_2'], window=loss_window, label='critic_loss_2', color='C1')
    plot_scores(log_dict['critic_loss'], window=loss_window, label='critic_loss', color='C0')
    plt.xlabel('step')
    plt.ylabel('critic loss')
    plt.legend()
    plt.subplot(1,3,3)

    # plot actor metrics
    lines = []
    plt.title('actor metrics')
    lines.append(plot_scores(log_dict['actor_loss'], window=loss_window, label='actor_loss'))
    plt.xlabel('step')
    plt.ylabel('actor_loss')
    if 'alpha' in log_dict.keys():
        plt.twinx()  # instantiate a second axes that shares the same x-axis
        lines.append(plot_scores(log_dict['alpha'], window=loss_window, label='alpha', color='C1'))
    plt.legend(lines, [line.get_label() for line in lines])

    plt.suptitle(title, fontsize=16)
    plt.savefig('results.png')
    plt.close()

def moving_average(a, n):
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).tolist()


def pad_and_get_mask(lists):
    """
    Pad a list of lists with zeros and return a mask of the same shape.
    """
    lens = [len(l) for l in lists]
    max_len = max(lens)
    arr = np.zeros((len(lists), max_len), float)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(lists)
    return np.ma.array(arr, mask=~mask)


def get_schedule(schedule: str):
    schedule_type = schedule.split('(')[0]
    schedule_args = schedule.split('(')[1].split(')')[0].split(',')
    if schedule_type == 'constant':
        assert len(schedule_args) == 1
        return lambda x: float(schedule_args[0])
    elif schedule_type == 'linear':
        assert len(schedule_args) == 4
        eps_max, eps_min, init_steps, anneal_steps = float(schedule_args[0]), float(schedule_args[1]), int(schedule_args[2]), int(schedule_args[3])
        return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - init_steps) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'cosine':
        assert len(schedule_args) == 4
        eps_max, eps_min, init_steps, anneal_steps = float(schedule_args[0]), float(schedule_args[1]), int(schedule_args[2]), int(schedule_args[3])
        return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - init_steps) / anneal_steps, 0, 1) * np.pi))
    else:
        raise ValueError('Unknown schedule: %s' % schedule_type)


def plot_scores(scores, steps=None, window=100, label=None, color=None):
    avg_scores = [moving_average(score, window) for score in scores]
    if steps is not None:
        for i in range(len(scores)):
            avg_scores[i] = np.interp(np.arange(steps[i][-1]), [0] + steps[i][-len(avg_scores[i]):], [0.0] + avg_scores[i])
    if len(scores) > 1:
        avg_scores = pad_and_get_mask(avg_scores)
        scores = avg_scores.mean(axis=0)
        scores_l = avg_scores.mean(axis=0) - avg_scores.std(axis=0)
        scores_h = avg_scores.mean(axis=0) + avg_scores.std(axis=0)
        idx = list(range(len(scores)))
        plt.fill_between(idx, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25, color=color)
    else:
        scores = avg_scores[0]
    plot, = plt.plot(scores, label=label, color=color)
    return plot


def merge_videos(video_dir):
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    videos = sorted(videos, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    videos = [VideoFileClip(video) for video in videos]
    clip = concatenate_videoclips([video for video in videos if video.duration > 0])
    os.makedirs('videos', exist_ok=True)
    clip.write_videofile(os.path.join('videos', f"{video_dir}.mp4"), verbose=False, logger=None)
    shutil.rmtree(video_dir)


def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_space_shape(space, is_vector_env=False):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, MultiDiscrete):
        return space.nvec[0]
    elif isinstance(space, Box):
        space_shape = space.shape[1:] if is_vector_env else space.shape
        if len(space_shape) == 1:
            return space_shape[0]
        else:
            return space_shape  # image observation
    elif isinstance(space, gym.spaces.dict.Dict):
        space_shape = space['observation'].shape[0]
        return space_shape
    else:
        print(type(space))
        raise ValueError(f"Space not supported: {space}")


def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ELU):
    sizes = [input_size] + list(layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)
