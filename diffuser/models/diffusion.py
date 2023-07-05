from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    extend,
    Losses,
)
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# Sample = namedtuple('Sample', 'trajectories values chains')
Sample = namedtuple('Sample', 'trajectories chains')

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)


    inds = torch.randperm(inds.shape[0], device=inds.device)

    
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, state_model, horizon, observation_dim, action_dim, predict_type, traj_len, ddim_timesteps=2, ddim=False, 
        n_timesteps=1000, clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, state_noise_start_t=3,
    ):
        super().__init__()
        self.horizon = horizon
        self.predict_type = predict_type
        self.traj_len = traj_len
        self.state_noise_start_t = state_noise_start_t
        if predict_type == 'obs_only':
            self.observation_dim = observation_dim
            self.action_dim = 0
            self.transition_dim = observation_dim
            self.loss_fn = Losses['l2']()
        elif predict_type == 'action_only':
            self.observation_dim = 0
            self.action_dim = action_dim
            self.transition_dim = action_dim
            self.loss_fn = Losses['l2']()
        elif predict_type == 'joint':
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.transition_dim = observation_dim + action_dim
            loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
            self.loss_fn = Losses['weightedl2'](loss_weights, self.action_dim)

        self.model = model
        self.state_model = state_model
        self.cnt = 0

        self.ddim = ddim
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.init_noise_sigma = 1.0

        step_ratio = n_timesteps // ddim_timesteps
        timesteps = (np.arange(0, ddim_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        prev_timesteps = (timesteps - step_ratio).copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        self.prev_timesteps = torch.from_numpy(prev_timesteps)
        self.step_ratio = step_ratio

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        ## log calculation clipped because the posterior variance
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, grad=None):
        if grad is not None:
            x_recon = self.predict_start_from_noise(x, t=t, noise=grad)
        else:
            noise_traj = self.model.sample(x,cond,t)
            noise_traj *= extract(self.sqrt_one_minus_alphas_cumprod, t, noise_traj.shape)
            # if (t<=self.state_noise_start_t).any():
            #     x_state = get_state_from_traj(x)
            #     t_state = t.repeat(self.traj_len)
            #     noise_state = self.state_model.sample(x_state, cond, t_state)
            #     noise_state = get_traj_from_state(x, self.traj_len)
            #     assert noise_state.shape == noise_traj.shape
            #     noise = 0.3 * noise_state + 0.7 * noise_traj
            # else:
            noise = 1.0 * noise_traj
            # pair_x = form_pairs(x)
            # noise_pair = self.pair_model.sample(pair_x, cond, torch.ones(pair_x.shape[0],device=pair_x.device)*t[0])
            # noise_pair = pair_to_traj(noise_pair, self.traj_len)
            # TODO: control the scale of two noises
            # noise = 0.8 * noise_traj + 0.2 * noise_state
            # noise = 0.9 * noise_traj + 0.1 * noise_pair
            # noise = 0.5 * noise_traj + 0.5 * noise_pair
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def q_ddim_posterior(self, x_start, x_t, t, prev_t, sigma_t):

        alpha_prod_t_prev = extract(self.alphas_cumprod, prev_t, x_t.shape) if (prev_t >= 0).any() else self.final_alpha_cumprod
        alpha_prod_t = extract(self.alphas_cumprod, t, x_t.shape)
        epsilon = (x_t - torch.sqrt(alpha_prod_t) * x_start) / torch.sqrt(1-alpha_prod_t)
        posterior_mean = torch.sqrt(1-alpha_prod_t_prev - sigma_t**2) * epsilon + torch.sqrt(alpha_prod_t_prev) * x_start 

        return posterior_mean

    @torch.no_grad()
    def p_mean_ddim_variance(self, x_t, cond, t, prev_t,sigma_t, grad=None, weight_traj=0.7):
        if grad is not None:
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=grad)
        else:
            # buffer = self.q_sample(x_start=cond[range(self.traj_len)], t=t)
            # noise_buffer = self.model.sample(buffer, cond, t)
            noise_traj = self.model.sample(x_t,cond,t)
            noise_traj *= extract(self.sqrt_one_minus_alphas_cumprod, t, noise_traj.shape)
            # if (t>=-1).any():
            #     noise = 1*noise_buffer + noise_traj*0
            # if self.cnt <= 3000:
            #     start_cond = (t<=self.state_noise_start_t).any() or (t>=self.n_timesteps-self.state_noise_start_t-30).any()
            # else:
            #     start_cond = False
            # if start_cond:
            #     x_state = get_state_from_traj(x_t)
            #     t_state = t.repeat(self.traj_len)
            #     noise_state = self.state_model.sample(x_state, cond, t_state)
            #     noise_state = get_traj_from_state(x_t, self.traj_len)
            #     assert noise_state.shape == noise_traj.shape

            #     x_recon_1 = self.predict_start_from_noise(x_t, t=t, noise=noise_traj)
            #     x_recon_2 = self.predict_start_from_noise(x_t, t=t, noise=noise_state)
            #     if self.clip_denoised:
            #         x_recon_1.clamp_(-1., 1.)
            #         x_recon_2.clamp_(-1., 1.)
            #     else:
            #         assert RuntimeError()
            #     x_recon = (1-weight_traj) * x_recon_2 + weight_traj * x_recon_1
            # else:
            noise = noise_traj
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=noise)
            x_recon.clamp_(-1., 1.)

        model_mean = self.q_ddim_posterior(x_start=x_recon, x_t=x_t, t=t, prev_t=prev_t, sigma_t=sigma_t)
        return model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, **sample_kwargs):
        
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        # x = pair_consistency(x)
        x = apply_conditioning(x, cond, self.action_dim)
        # x = new_apply_conditioning(x, cond, self.action_dim)
        x = self.to_torch(x)
        chain = [x] if return_chain else None

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x = self.n_step_guided_p_sample(x, cond, t, **sample_kwargs)
            # x = new_apply_conditioning(x, cond, self.action_dim)
            x = apply_conditioning(x, cond, self.action_dim)
            # x = pair_consistency(x)
            if return_chain: chain.append(x)

        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, chain)
 
    @torch.no_grad()
    def ddim_sample_loop(self, shape, cond, verbose=True, return_chain=False, **sample_kwargs):
        # calculations for diffusion q(x_t | x_{t-1}) and others
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = extend(x, cond)
        x = apply_conditioning(x, cond, self.action_dim)
        x = self.to_torch(x)
        chain = [x] if return_chain else None
        for t, prev_t in zip(self.timesteps, self.prev_timesteps):
            t = make_timesteps(batch_size, t, device)
            prev_t = make_timesteps(batch_size, prev_t, device)
            x = self.n_step_guided_ddim_sample(x, cond, t, prev_t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)
            if return_chain: chain.append(x)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x,chain)

    @torch.no_grad()
    def n_step_guided_ddim_sample(
        self, x, cond, t, prev_t, eta=0, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=False):

        alpha_prod_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_prod_t_prev = extract(self.alphas_cumprod, prev_t, x.shape) if (prev_t >= 0).any() else torch.ones_like(alpha_prod_t) * self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        model_std = eta * variance ** (0.5)

        model_mean = self.p_mean_ddim_variance(x_t=x, cond=cond, t=t,prev_t=prev_t, sigma_t=model_std, grad=None)
        # no noise when t == 0
        if (model_std == 0).all():
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + model_std * noise

    @torch.no_grad()
    def n_step_guided_p_sample(self, x, cond, t, eta=0, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True):
    
        model_log_variance = extract(self.posterior_log_variance_clipped.to(t.device), t, x.shape)
        model_var = torch.exp(model_log_variance)
        
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)

        # no noise when t == 0
        noise = torch.randn_like(x)
        b = x.shape[0]
        nonzero_mask = (1-(t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # noise[t == 0] = 0

        # return model_mean + model_std * noise, y
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        # return model_mean + model_std * noise
   
    
    def to_torch(self, x_in):
        if type(x_in) is dict:
            return {k: self.to_torch(v) for k, v in x_in.items()}
        elif torch.is_tensor(x_in):
            return x_in.to(self.betas.device)
        return torch.tensor(x_in, device=self.betas.device)

    @torch.no_grad()
    def conditional_sample(self, cond, train_ddim=None, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        if not train_ddim:
            self.sample_kwargs = sample_kwargs
        horizon = horizon or self.horizon
        # if horizon == 2:
        #     shape = (self.traj_len, horizon, self.transition_dim)
        # else:
        #     shape = (len(cond[0]), horizon, self.transition_dim)

        # shape = (len(cond[0]), max(cond.keys())+1, self.transition_dim)
        shape = (len(cond[0]), max(cond.keys())+1, self.transition_dim//2)
        if train_ddim:
            return self.ddim_sample_loop(shape, cond, **self.sample_kwargs)
        if train_ddim == False:
            return self.p_sample_loop(shape, cond, **self.sample_kwargs)
        if self.ddim:
            return self.ddim_sample_loop(shape, cond, **self.sample_kwargs)
        else:
            return self.p_sample_loop(shape, cond, **self.sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample


    def p_losses(self, x_start, cond, t):

        # Train trajectory model
        x_start = extend(x_start, cond)
        noise = torch.randn_like(x_start)
        # mask = ~(x_start.sum(-1) == 0)[:,:,None]
        # noise = noise * mask
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if x_noisy.shape[1] != 1:
            x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
            # x_noisy = x_noisy * mask
            x_recon = self.model(x_noisy, cond, t)
        else:
            x_recon = self.state_model(x_noisy, cond, t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            x_recon *= extract(self.sqrt_one_minus_alphas_cumprod, t, x_recon.shape)
            loss, info = self.loss_fn(x_recon, noise)
        else:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)
            loss, info = self.loss_fn(x_recon, x_start)


        # Train pair model
        # x_start_state = get_state_from_traj(x_start)
        # noise_state = get_state_from_traj(noise_traj)
        # t_state = t[:,None].repeat((1, x_noisy_traj.shape[1])).flatten()
        # x_noisy_state = self.q_sample(x_start=x_start_state, t=t_state, noise=noise_state)
        # x_recon_state = self.state_model(x_noisy_state, cond, t_state)
        # assert noise_state.shape == x_recon_state.shape
        # if self.predict_epsilon:
        #     loss2, info2 = self.loss_fn(x_recon_state, noise_state)
        # else:
        #     loss2, info2 = self.loss_fn(x_recon_state, x_start_state)

        # x_start_pair = form_pairs(x_start)
        # idx = torch.randint(0, x_start_pair.shape[0], (self.pair_batch_size,))
        # x_start_pair = x_start_pair[idx]
        # noise_pair = torch.randn_like(x_start_pair)
        # noise_pair = form_pairs(noise_traj)
        # t_pair = t[:,None].repeat((1, x_noisy_traj.shape[1]-1)).flatten()
        # x_noisy_pair = self.q_sample(x_start=x_start_pair, t=t_pair, noise=noise_pair)
        # TODO: add condition here
        # x_noisy_pair[:,0,:] = x_start_pair[:,0,:]
        # x_recon_pair = self.pair_model(x_noisy_pair, cond, t_pair)
        # assert noise_pair.shape == x_recon_pair.shape
        # if self.predict_epsilon:
        #     loss2, info2 = self.loss_fn(x_recon_pair, noise_pair)
        # else:
        #     loss2, info2 = self.loss_fn(x_recon_pair, x_start_pair)

        # Train joint model
        # x_recon_pair_to_traj = pair_to_traj(x_recon_pair, self.traj_len)
        # x_recon_joint = x_recon_pair_to_traj
        # loss3, info3 = self.loss_fn(x_recon_joint, noise_traj)

        return loss, info    # only backward once loss3

    def loss(self, x, cond):

        # t = torch.randint(1, self.ddim_timesteps+1, (batch_size,), device=x.device).long()
        # t *= self.n_timesteps // self.ddim_timesteps
        # t -= 1
        # unpacked_x = pad_packed_sequence(x, batch_first=True)[0]
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # x = x.to(torch.float)
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

