import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, predict_type, keys=None):

        if predict_type == 'obs_only':
            self.observation_dim = dataset.observation_dim
            self.action_dim = None
            keys = ['observations']
        elif predict_type == 'action_only':
            self.action_dim = dataset.action_dim
            self.observation_dim = None
            keys = ['actions']
        elif predict_type == 'joint':
            self.observation_dim = dataset.observation_dim
            self.action_dim = dataset.action_dim
            keys = ['observations','actions']
        
        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key in keys:  
            if key == 'actions':
                val = np.array((list(np.ones(self.action_dim)*(1.)),list(np.ones(self.action_dim)*(-1.))),dtype=np.float32)
                self.normalizers[key] = normalizer(val)
            elif key == 'observations':
                # val = np.array(([ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ],[ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]),dtype=np.float32)
                val = np.array(([ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ],[ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]),dtype=np.float32)
                self.normalizers[key] = normalizer(val)
            else:
                raise NotImplementedError


    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened



#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        #obs_mins: [ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ]
        #obs_maxs: [ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()



class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        try:
            x = (x - self.mins) / (self.maxs - self.mins)
        except:
            mins = np.tile(self.mins,2)
            maxs = np.tile(self.maxs,2)
            x = (x - mins) / (maxs - mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            # raise ValueError('Wrong!')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        try:
            return x * (self.maxs - self.mins) + self.mins
        except:
            mins = np.tile(self.mins,2)
            maxs = np.tile(self.maxs,2)
            return x * (maxs - mins) + mins

