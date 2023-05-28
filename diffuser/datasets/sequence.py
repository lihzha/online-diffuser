from collections import namedtuple
import numpy as np
import torch
import pdb
import copy

from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env, predict_type, horizon=64,
        normalizer='LimitsNormalizer', max_path_length=1000,
        max_n_episodes=10000):

        self.env = env
        self.horizon = horizon
        self.action_dim = self.env.action_space.shape[0]
        self.observation_dim = self.env.observation_space['observation'].shape[0]
        self.max_path_length = max_path_length
        self.max_n_episodes = max_n_episodes

        self.normalizer_name = normalizer
        self.predict_type = predict_type
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty=None)
        self.fields = fields
        self.normalizer = DatasetNormalizer(self, normalizer, predict_type)
        
        print(fields)

    def set_fields(self, fields):
        self.fields = copy.deepcopy(fields)
        if self.horizon != 1:
            self.fields.remove_short_episodes(self.horizon)
            self.fields._count = len(self.fields['path_lengths'])
        self.fields.finalize()
        self.fields.observation_dim = self.observation_dim
        self.fields.action_dim = self.action_dim
        self.n_episodes = self.fields.n_episodes
        self.path_lengths = self.fields['path_lengths']
        self.normalize()
        

    def normalize(self, keys=None):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        if self.predict_type == 'obs_only':
            keys=['observations']
        elif self.predict_type == 'action_only':
            keys=['actions']
        elif self.predict_type == 'joint':
            keys=['observations', 'actions']
        else:
            raise NotImplementedError
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
            max_start = min(path_length - horizon + 1, self.max_path_length - horizon + 1)
            if max_start < 0:
                max_start = 0
            if max_start > 0:
                for start in range(max_start):
                    end = start + horizon
                    indices.append((i, start, end))
            elif max_start == 0:
                end = horizon
                indices.append((i,0,end))
            else:
                pass
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

        if self.predict_type == 'joint':
            observations = self.fields.normed_observations[path_ind, start:end]
            actions = self.fields.normed_actions[path_ind, start:end]
            trajectories = np.concatenate([actions, observations], axis=-1)
        elif self.predict_type == 'obs_only':
            observations = self.fields.normed_observations[path_ind, start:end]
            trajectories = observations
        elif self.predict_type == 'action_only':
            actions = self.fields.normed_actions[path_ind, start:end]
            trajectories = actions
        conditions = self.get_conditions(observations)
        batch = Batch(trajectories, conditions)
        return batch
    
    def get_obs(self, idx_set, indices, eps=1e-4):
        first = True
        for idx in idx_set:
            path_ind, start, end = indices[idx]
            if self.predict_type == 'joint':
                observations = self.fields.normed_observations[path_ind, start:end]
                actions = self.fields.normed_actions[path_ind, start:end]
                trajectories = np.concatenate([actions, observations], axis=-1)
            elif self.predict_type == 'obs_only':
                observations = self.fields.normed_observations[path_ind, start:end]
                trajectories = observations
            elif self.predict_type == 'action_only':
                actions = self.fields.normed_actions[path_ind, start:end]
                trajectories = actions
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
