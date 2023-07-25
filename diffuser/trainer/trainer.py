import os
import copy
import numpy as np
import torch
import einops
import pdb

from ..utils.arrays import new_batch_to_device, to_np, to_device, apply_dict
from ..utils.timer import Timer
import torch.nn as nn
import matplotlib.pyplot as plt
# import cv2
# from torch.nn.utils.rnn import pack_padded_sequence

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        model,
        dataset,
        dataset_fake,
        device,
        renderer,
        train_batch_size,
        results_folder,
        loadpath,
        ema_decay=0.995,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        # update_ema_every=1000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.renderer = renderer
        self.ema = EMA(ema_decay)
        self.model = model
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataset_fake = dataset_fake
            
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
        
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.device = device

        self.reset_parameters()
        self.step = 0
        if loadpath:
            self.load(loadpath)


    def create_dataloader(self, use_fake_buffer=False, batch_size=None, sampler=None):
        # def my_collate(batch):
        #     traj = torch.as_tensor([b.trajectories for b in batch], dtype=torch.float32)
        #     path_length = torch.as_tensor([b.path_lengths for b in batch],dtype=torch.int)
        #     packed_traj = pack_padded_sequence(traj, path_length, batch_first=True, enforce_sorted=False)
        #     conditions = {key: torch.as_tensor([b.conditions[key] for b in batch], dtype=torch.float32) for key in batch[0].conditions.keys()}
            # return (packed_traj, conditions)
        def my_collate(batch):
            assert len(batch) == 1
            traj = batch[0].trajectories
            cond = batch[0].conditions
            return (traj, cond)
        if batch_size == None and sampler == None:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=my_collate))
        elif batch_size != None and sampler == None:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True))
        elif sampler != None:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True, sampler=sampler))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True))
        if use_fake_buffer:
            if batch_size == None and sampler == None:
                self.dataloader_fake = cycle(torch.utils.data.DataLoader(
                        self.dataset_fake, batch_size=self.batch_size, num_workers=1, shuffle=True, pin_memory=True, collate_fn=my_collate))
            elif batch_size != None and sampler == None:
                self.dataloader_fake = cycle(torch.utils.data.DataLoader(
                        self.dataset_fake, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True))
            elif sampler != None:
                self.dataloader_fake = cycle(torch.utils.data.DataLoader(
                        self.dataset_fake, batch_size=self.batch_size, num_workers=1, pin_memory=True, sampler=sampler))
            self.dataloader_fake_vis = cycle(torch.utils.data.DataLoader(
                        self.dataset_fake, batch_size=1, num_workers=0, shuffle=True, pin_memory=True))


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, use_fake_buffer=False):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):

                batch = next(self.dataloader)
                batch = new_batch_to_device(batch, device=self.device)

                loss, infos = self.diffusion_model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                if use_fake_buffer:
                    batch = next(self.dataloader_fake)
                    batch = new_batch_to_device(batch, device=self.device)

                    loss, infos = self.diffusion_model.loss(*batch, fake=True)
                    loss = loss / self.gradient_accumulate_every / 2
                    loss.backward()
                # TODO: what is max_norm?
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            self.optimizer.step()
            self.optimizer.zero_grad()


            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_reference(self.n_reference)
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')



    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        if isinstance(epoch,str):
            loadpath = epoch
        else:
            loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        # normed_observations = trajectories[:, :, self.dataset.action_dim:]
        normed_observations = trajectories
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        observations = observations.squeeze(0)

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations, ncol=1)
        # render_idx = np.random.choice(observations.shape[0], 9, replace=False)
        # fig = plt.figure(figsize=(12, 12))
        # img_idx = 331
        # for i in render_idx: 
        #     ax = fig.add_subplot(img_idx,projection='3d')
        #     img_idx += 1
        #     ax.plot3D(observations[i,:,0], observations[i,:,1],observations[i,:,2])
    
        # plt.savefig(savepath)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        n_samples = 10

        for i in range(3):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = batch.conditions
            trajectories = batch.trajectories
            trajectories = trajectories.squeeze(0)
            if i == 0:
                conditions[0] = np.array([1,1,0,0])[None]
                conditions[trajectories.shape[1]-1] = np.array([1,8,0,0])[None]

            elif i == 1:
                conditions[0] = np.array([6,1.8,0,0])[None]
                conditions[trajectories.shape[1]-1] = np.array([3.,3.,0.,0.])[None]
            
            elif i == 2:
                conditions[0] = np.array([1.,8.,0,0])[None]
                conditions[trajectories.shape[1]-1] = np.array([5.,8.,0.,0.])[None]

            conditions[0] = self.dataset.normalizer.normalize(conditions[0], 'observations')
            conditions[trajectories.shape[1]-1] = self.dataset.normalizer.normalize(conditions[trajectories.shape[1]-1], 'observations')
            conditions[0] = torch.tensor(conditions[0])
            conditions[trajectories.shape[1]-1] = torch.tensor(conditions[trajectories.shape[1]-1])

            conditions = to_device(conditions, self.device)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.diffusion_model.conditional_sample(conditions, train_ddim=True)
            samples = to_np(samples.trajectories)
            # samples = np.concatenate((samples[0],samples[1,1:]),axis=0)[None]
            normed_observations = samples

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            if self.diffusion_model.condition_type == 'extend':
                round = 2
            else:
                round = 1
            for j in range(round):
                obs = observations[:,:,j*4:(j+1)*4]

                savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}-{j}.png')
                self.renderer.composite(savepath, obs,ncol=5)
    

    def render_buffer(self, batchsize, obs):
        
        savepath = os.path.join(self.logdir, f'sample_reference-{self.step}.png')
        # obs_num = obs.shape[0]  
        obs_num = len(obs.keys())

        idx = np.random.choice(obs_num, batchsize, replace=True)

        self.renderer.composite(savepath, obs[idx], ncol=5)
