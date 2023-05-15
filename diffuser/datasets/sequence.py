from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
import gymnasium as gym
import panda_gym

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, online=False, predict_action=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        print(env)
        # self.env = env = load_environment(env)
        self.env = gym.make("PandaReach-v3", render_mode="rgb_array")
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.max_n_episodes = max_n_episodes
        self.termination_penalty = termination_penalty
        self.normalizer_name = normalizer
        self.predict_action = predict_action
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        self.fields = fields
        # if online == False:
        #     itr = sequence_dataset(env, self.preprocess_fn)
        #     for i, episode in enumerate(itr):
        #         if i == max_n_episodes:
        #             break
        #         fields.add_path(episode)
        #     fields.finalize()

        #     self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        #     self.indices = self.make_indices(fields.path_lengths, horizon)
        #     self.path_lengths = fields.path_lengths
        #     self.observation_dim = fields.observations.shape[-1]
        #     self.action_dim = fields.actions.shape[-1]
        #     self.n_episodes = fields.n_episodes
        #     self.fields = fields
        #     self.normalize()
        
        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        if not self.predict_action:
            keys=['observations']
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            if max_start != 0:
                for start in range(max_start):
                    end = start + horizon
                    indices.append((i, start, end))
            else:
                end = horizon
                indices.append((i,0,end))
        indices = np.array(indices)
        return indices

    def make_new_indices(self, path_lengths, horizon):
        '''
            makes non-overlapping indices for computing energy;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            if max_start != 0:
                for start in range(0,max_start,horizon):
                    end = start + horizon
                    indices.append((i, start, end)) 
                if end < path_length-1:
                    indices.append((i, path_length-horizon, path_length)) 
            else:
                end = horizon
                indices.append((i,0,end))
        indices = np.array(indices)
        return indices


    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        if self.predict_action:
            actions = self.fields.normed_actions[path_ind, start:end]
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations
        conditions = self.get_conditions(observations)
        batch = Batch(trajectories, conditions)
        return batch
    
    def get_obs(self, idx_set, indices, eps=1e-4):
        first = True
        for idx in idx_set:
            path_ind, start, end = indices[idx]
            observations = self.fields.normed_observations[path_ind, start:end]
            if self.predict_action:
                actions = self.fields.normed_actions[path_ind, start:end]
                trajectories = np.concatenate([actions, observations], axis=-1)
            else:
                trajectories = observations
            if first:
                traj = trajectories[None,:]
                first = False
            else:
                traj = np.concatenate((traj, trajectories[None,:]), axis=0)
        return traj

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True
            
    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
