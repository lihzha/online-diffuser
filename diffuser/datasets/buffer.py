import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty, batch_size=32):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int),
        }
        # self._dict = {}
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.path_idx = {}
        self.batch_size = batch_size

    def __repr__(self):
        try:
            return '[ datasets/buffer ] Fields:\n' + f"{self._count} episodes\n" + str(len(self._dict['index'])) + ' index'
        except:
            return '[ datasets/buffer ] Fields:\n'

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())
        self.keys.append('conditions')
        self._dict['conditions'] = np.zeros((self.max_n_episodes, path['observations'][0].shape[-1]))

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths' and k!= 'index'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path, cond):
        path_length = len(path['observations'])
        if path_length > self.max_path_length:
            path_length = self.max_path_length
        # assert path_length <= self.max_path_length
        for key in path.keys():
            path[key] = path[key][:self.max_path_length]
        ## if first path added, set keys based on contents
        self._add_keys(path)
        self._dict['conditions'][self._count] = cond
        ## add tracked keys in path
        for key in self.keys:
            if key == 'conditions':
                continue
            array = atleast_2d(path[key])
            # if self._count % self.max_n_episodes == 0 and self._count > 0: 
            if key not in self._dict: self._allocate(key, array)
            #     self.expand_dict(key, array)
            self._dict[key][self._count, :path_length] = array
            self._dict[key][self._count, path_length:] = 0                    

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            # assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        # if self._count % self.max_n_episodes == 0 and self._count > 0: 
            # self._dict['path_lengths'] = np.concatenate((self._dict['path_lengths'], np.zeros(self.max_n_episodes,dtype=np.int32)))
        self._dict['path_lengths'][self._count] = path_length
        ## increment path counter
        self._count += 1
        self._count = self._count % self.max_n_episodes
        
    # def add_path(self, path ,length=None):

    #     path_length = len(path['observations']) if length is None else length
    #     ## if first path added, set keys based on contents
    #     self._add_keys(path)
    #     first = self.update_idx(path, path_length)
    #     path_idx = self.path_idx[path_length]

    #     if first:
    #         for key in self.keys:
    #             if key == 'index':
    #                 continue
    #             if key not in self._dict:
    #                 self._dict[key] = {path_idx: path[key][None]}
    #             else:
    #                 self._dict[key][path_idx] = path[key][None]
    #     else:
    #         for key in self.keys:
    #             if key == 'index':
    #                 continue
    #             self._dict[key][path_idx] = np.concatenate((self._dict[key][path_idx], path[key][None]),axis=0)
    #         if self._dict['observations'][path_idx].shape[0] == self.batch_size:
    #             self.path_idx[path_length] *= 10

    #     self._count += 1
    #     self._count = self._count % self.max_n_episodes


    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

    # def finalize(self):
    #     self._add_attributes()
    #     print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

    def expand_dict(self, key, array):

        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        expand = np.zeros(shape, dtype=np.float32)
        self._dict[key] = np.concatenate((self._dict[key],expand),axis=0)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def reinit(self, data):
        
        for i in range(data.shape[0]):
            episode = {}
            observations = data[i,:,4:]
            episode['observations'] = observations
            actions = data[i,:,:4]
            episode['actions'] = actions
            episode['terminals'] = np.zeros((observations.shape[0],1))
            self.add_path(episode)

        for i in range(data.shape[0]):
           self._dict['path_lengths'][i] = self._dict['observations'][i,:,0].nonzero()[0][-1] + 1
        print(self)

    def remove_short_episodes(self, max_path_length):

        path_length = self._dict['path_lengths']
        mask = path_length >= max_path_length
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][mask]

        
    def update_idx(self, path, path_length):

        if path_length not in self.path_idx.keys():
            self.path_idx[path_length] = path_length
        

        path['index'] = self.path_idx[path_length]

        if 'index' not in self._dict:
            self._dict['index'] = []

        if path['index'] not in self._dict['index']:
            self._dict['index'].append(path['index'])
            return True
        else:
            return False