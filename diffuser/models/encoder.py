import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import torch.nn.functional as F
import numpy as np


features = 16
# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, obs_dim):
        super(LinearVAE, self).__init__()

        self.obs_dim = obs_dim
        # encoder
        self.enc1 = nn.Linear(in_features=obs_dim, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=1024)
        self.enc3 = nn.Linear(in_features=1024, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=1024)
        self.dec3 = nn.Linear(in_features=1024, out_features=obs_dim)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        mu, log_var = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        reconstruction = self.decode(z).view(-1, self.obs_dim)
        return reconstruction, mu, log_var

    def encode(self, x):

        """x.shape -> (batch_size, obs_dim)"""

        x = F.mish(self.enc1(x))
        x = F.mish(self.enc2(x))
        x = self.enc3(x).view(-1, 2, features)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        return mu, log_var

    def decode(self, z):
        x = F.mish(self.dec1(z))
        x = F.mish(self.dec2(x))
        reconstruction = F.mish(self.dec3(x))
        return reconstruction
    

class q_goal:

    def __init__(self, model, dataset, alpha = -1, train_lr = 2e-5):
        
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
        self.alpha = alpha
        self.device = model.device
        self.loss_fn = nn.MSELoss()
        self.dataset = dataset

    def vae_training(self, batch):

        """batch.shape -> (batch_size, obs_dim)"""

        x_recon, mu, log_var = self.model(batch)
        loss = self.vae_loss(x_recon, batch, log_var, mu)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def vae_loss(self, x_recon, x, log_var, mu):

        mse = F.mse_loss(x_recon, x)
        kld = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) -1. - log_var)
        assert mse.shape == kld.shape, 'Wrong loss function!'
        loss = mse + kld
        return loss
    
    def cal_p_goal(self, batch_si):
        
        mu, log_var = self.model.encode(batch_si)
        p_zx = 

    def cal_omega(self, batch):

        """batch.shape -> (batch_size, obs_dim)"""
        omega_list = []
        for si in batch[:]:
            omega_list.append(self.model(si) ** self.alpha)
        return omega_list

    def cal_p_skewed(self, batch_si, rand_batch):

        omega_list = self.cal_omega(batch_si)
        norm_cst = sum(omega_list)
        p_skewed_nonzero = self.model(batch_si)/norm_cst*omega_list
        p_skewed_zero = torch.zeros_like(rand_batch)
        p_skewed = torch.cat((p_skewed_nonzero, p_skewed_zero),axis=1)
        return p_skewed

    def mle_loss(self, b_si, rand_batch):

        train_batch = torch.cat((b_si, rand_batch),axis=1)
        p_si = self.model(train_batch)
        p_skewed = self.cal_p_skewed(b_si, rand_batch)
        loss = self.loss_fn(p_si, p_skewed)
        return loss

    def train(self, epoch, batchsize):

        for _ in range(epoch):
            batch_si = self.get_batch(batch_size=batchsize)
            rand_batch = torch.randn_like(batch_si)
            loss = self.loss(batch_si, rand_batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_batch(self, batch_size):

        _dict = self.dataset._dict
        si = _dict['observations']
        sample_index = np.random.choice(si.shape[0],size=batch_size,replace=True)
        return si[sample_index]

    def sample(self):

        z = torch.randn(6, device=self.device)
        target = self.model.decoding(z)
        return target
