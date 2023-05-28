import os
import copy
import numpy as np
import torch
import einops
import pdb

from ..utils.arrays import batch_to_device, to_np, to_device, apply_dict
from ..utils.timer import Timer
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

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


    def create_dataloader(self, batch_size=None):
        if batch_size == None:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=True, pin_memory=True))
        else:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                    self.dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True))

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

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                if step==0:
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device)

                loss, infos = self.diffusion_model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
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
                # self.render_reference(self.n_reference)
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
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
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

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        render_idx = np.random.choice(observations.shape[0], 9, replace=False)
        fig = plt.figure(figsize=(12, 12))
        img_idx = 331
        for i in render_idx: 
            ax = fig.add_subplot(img_idx,projection='3d')
            img_idx += 1
            ax.plot3D(observations[i,:,0], observations[i,:,1],observations[i,:,2])
    
        plt.savefig(savepath)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        n_samples = 10
        batch_size = 1
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = batch.conditions
            conditions[0] = np.array([1,1,0,0])[None]
            conditions[batch.trajectories.shape[1]-1] = np.array([1,8,0,0])[None]
            conditions[0] = self.dataset.normalizer.normalize(conditions[0], 'observations')
            conditions[batch.trajectories.shape[1]-1] = self.dataset.normalizer.normalize(conditions[batch.trajectories.shape[1]-1], 'observations')
            conditions[0] = torch.tensor(conditions[0])
            conditions[batch.trajectories.shape[1]-1] = torch.tensor(conditions[batch.trajectories.shape[1]-1])
            conditions = to_device(batch.conditions, self.device)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.diffusion_model.conditional_sample(conditions, train_ddim=True)
            samples = to_np(samples.trajectories)
            normed_observations = samples

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            # observations = observations[:,0,:][None]
            # if i == 0:
            #     obs = observations
            # else:
            #     obs = np.concatenate((obs,observations))
        # observations = observations.reshape((n_samples, 640, 4))
        savepath = os.path.join(self.logdir, f'sample-{self.step}-{0}.png')
        self.renderer.composite(savepath, observations,ncol=5)
    

    def render_buffer(self, batchsize, obs):
        
        savepath = os.path.join(self.logdir, f'sample_reference-{self.step}.png')
        obs_num = obs.shape[0]     
        idx = np.random.choice(obs_num, batchsize, replace=True)

        self.renderer.composite(savepath, obs[idx], ncol=5)
