import torch
import numpy as np
from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.sampling.guides import EBM_DensityGuide


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, p_explore=0):
    
    model_log_variance = extract(model.posterior_log_variance_clipped.to(t.device), t, x.shape)
    model_var = torch.exp(model_log_variance)
    explore = np.random.binomial(1,p_explore)
    if explore:
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                x_0 = model.reverse_q_sample(x,t)
                x_0 = apply_conditioning(x_0, cond, model.action_dim)
                prob, grad = guide.gradients(x_0)

            if scale_grad_by_std:
                grad = model_var * grad
            
            grad_zero_mask = torch.zeros_like(grad)
            grad_zero_mask[:,model.horizon//2:,:] = 1 

            x = x - scale * grad * grad_zero_mask
            x = apply_conditioning(x, cond, model.action_dim)
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    b = x.shape[0]
    nonzero_mask = (1-(t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    # noise[t == 0] = 0

    # return model_mean + model_std * noise, y
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    # return model_mean + model_std * noise


@torch.no_grad()
def n_step_guided_ddim_sample(
    model : GaussianDiffusion, x, cond, t, next_t, guide, eta=0, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=False, p_explore=0):

    alpha = extract(model.alphas_cumprod_ddim, t+1, x.shape)
    alpha_next = extract(model.alphas_cumprod_ddim, next_t+1, x.shape)
    std_const = torch.sqrt((1-alpha_next)/(1-alpha)*(1-alpha/alpha_next))
    model_std = eta * std_const
    model_var = model_std ** 2
    explore = np.random.binomial(1,p_explore)
    if explore:
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                if isinstance(guide, EBM_DensityGuide):
                    grad, guidance = guide.gradients(x, cond, t)
                else:
                    prob, grad = guide.gradients(x)
                if scale_grad_by_std:
                    guidance = model_var * guidance
                
                # guidance_zero_mask = torch.zeros_like(guidance)
                # grad_zero_mask[:,model.horizon//2:,:] = 1 
                # guidance_zero_mask[:,:,:] = 1 

                if isinstance(guide, EBM_DensityGuide):
                    guidance = (scale * guidance).clamp(-1,1)
                    x = x + guidance
                else:
                    x = x - scale * grad
            x = apply_conditioning(x, cond, model.action_dim)
    else:
        grad = None

    if isinstance(guide, EBM_DensityGuide):
        model_mean= model.p_mean_ddim_variance(x=x, cond=cond, t=t,next_t=next_t, sigma_t=model_std, grad=grad)
    else:
        model_mean= model.p_mean_ddim_variance(x=x, cond=cond, t=t,next_t=next_t, sigma_t=model_std, grad=None)

    # no noise when t == 0
    if (model_std == 0).all():
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + model_std * noise
   