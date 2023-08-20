import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            # nn.Mish(),
            nn.SiLU(),
        )
        # self.block1 = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        # self.norm = nn.GroupNorm(n_groups, out_channels)
        # self.silu = nn.SiLU()

    def forward(self, x):
        # x = self.block1(x)[:,:,None,:]
        # x = self.norm(x).squeeze(2)
        # x = self.silu(x)
        # return x
        return self.block(x)


#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def extend(x, conditions):
    cond_array = torch.ones_like(x)
    traj_len = cond_array.shape[1]
    assert len(list(conditions.keys())) == 2
    for t, val in conditions.items():
        assert len(val.shape) == 2
        if t == 0:
            # cond_array[:,::2, :] *= val[0]
            cond_array[:,:traj_len, :] *= val[0]
        else:
            # cond_array[:, 1::2, :] *= val[0]
            cond_array[:,traj_len:, :] *= val[0]
    x = torch.cat((cond_array, x), dim=2)
    return x

def apply_conditioning(x, conditions, action_dim, condition_type):
    if condition_type == 'extend':
        for t, val in conditions.items():
            x[:, t, action_dim+4:] = val.clone()
    else:
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()

    return x

def new_apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        if t>0:
            x[t, 1, action_dim:] = val.clone()
        else:
            x[t, 0, action_dim:] = val.clone()
    return x

def pair_consistency(x):
    shape = x.shape
    x[1:,0] = x[:shape[0]-1, -1]
    return x

@torch.no_grad()
def is_pair(x):
    shape = x.shape
    assert len(shape) == 3
    if torch.norm(x[1:,0]-x[:shape[0]-1, -1]) < 1e-4:
        return True
    else:
        return False
    
def form_pairs(x):
    shape = x.shape
    if shape[0] == 1:
        new_x = torch.cat((x[0,:shape[1]-1,:][:,None],x[0,1:shape[1],:][:,None]),dim=1)
        new_x.requires_grad_(x.requires_grad)
    else:
        traj_len = x.shape[1]
        new_x = x.reshape((-1,shape[-1]))
        shape = new_x.shape
        new_x = torch.cat((new_x[:shape[0]-1,:][:,None],new_x[1:shape[0],:][:,None]),dim=1)
        idx = torch.linspace(0, new_x.shape[0]-1, (new_x.shape[0]), dtype=torch.int)
        idx = idx[idx%traj_len!=traj_len-1]
        new_x = new_x[idx]
        new_x.requires_grad_(x.requires_grad)
    return new_x

def pair_to_traj(x, traj_len):
    new_x = x[:,0,:].reshape((-1,traj_len-1,x.shape[-1]))
    idx = torch.linspace(0,x.shape[0]-1,x.shape[0],dtype=torch.int)
    idx = idx[idx % (traj_len-1) == (traj_len-2)]
    last = x[idx,1,:][:,None]
    assert len(last.shape)==3 and last.shape[1]==1
    return torch.cat((new_x,last),dim=1)

def get_state_from_traj(x):
    return x.reshape((-1, x.shape[-1]))[:,None]

def get_traj_from_state(x, traj_len):
    return x.reshape((-1, traj_len, x.shape[-1]))

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class UnweightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        loss = loss.mean()
    
        return loss, {'loss':loss}


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class UnweightedL1(UnweightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class UnweightedL2(UnweightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'weightedl1': WeightedL1,
    'weightedl2': WeightedL2,
    'l1': UnweightedL1,
    'l2': UnweightedL2,
}
