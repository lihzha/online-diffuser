from collections import namedtuple
import torch
import einops
import pdb
import numpy as np
import diffuser.utils as utils

# Trajectories = namedtuple('Trajectories', 'actions observations values')
Trajectories = namedtuple('Trajectories', 'actions observations')
Trajectories_obs = namedtuple('Trajectories', 'observations')

class GuidedPolicy:

    def __init__(self, diffusion_model, normalizer, predict_type, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.predict_type = predict_type
        sample_kwargs.__delitem__('_device')
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):

        conditions = self._format_conditions(conditions, batch_size)
        # if len(conditions[0].shape) > 1:
        #     conditions = {k: self.normalizer.normalize(v, 'observations') for k, v in conditions.items()}
        # else:
        #     conditions = {k: self.normalizer.normalize(v, 'observations')[None] for k, v in conditions.items()}
        # conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)
        # samples = self.diffusion_model(conditions)
        trajectories = utils.to_np(samples.trajectories)
        if self.diffusion_model.condition_type == 'extend':
            trajectories = trajectories[:,:,self.diffusion_model.transition_dim//2:]
        # trajectories = samples.detach().cpu().numpy()
        # trajectories = trajectories.squeeze()
        if self.predict_type == 'joint':
            normed_observations = trajectories[:, :, self.action_dim:]
            normed_actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(normed_actions, 'actions')
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
            trajectories = Trajectories(actions, observations)
            return trajectories
        elif self.predict_type == 'obs_only':
            normed_observations = trajectories[:, :, self.action_dim:]
            normed_observations = self.pick_closest_to_goal(normed_observations, conditions)
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
            trajectories = Trajectories_obs(observations)
            return trajectories
        elif self.predict_type == 'action_only':
            raise NotImplementedError

    def pick_closest_to_goal(self, obs, cond):
        dist = ((obs[:,-2] - cond[max(cond.keys())].cpu().numpy()) ** 2).sum(-1)
        return obs[np.argmax(np.array(dist))]


    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions =  {
            k: v.squeeze()
            for k, v in conditions.items()
        }
        # conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        return conditions
