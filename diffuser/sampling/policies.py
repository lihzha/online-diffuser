from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

# Trajectories = namedtuple('Trajectories', 'actions observations values')
Trajectories = namedtuple('Trajectories', 'actions observations')
Trajectories_obs = namedtuple('Trajectories', 'observations')

class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, predict_action, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.predict_action = predict_action
        sample_kwargs.__delitem__('_device')
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, it=1, batch_size=1, verbose=True, p_explore=0):

        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size, it)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions,  verbose=verbose, guide=self.guide, p_explore=p_explore, **self.sample_kwargs)
        # samples = self.diffusion_model(conditions)
        trajectories = utils.to_np(samples.trajectories)
        # trajectories = samples.detach().cpu().numpy()
        # trajectories = trajectories.squeeze()
        if self.predict_action:
            normed_observations = trajectories[:, :, self.action_dim:]
            normed_actions = trajectories[:, :, :self.action_dim]
            if it:
                actions = self.normalizer.unnormalize(actions, 'actions')
                observations = self.normalizer.unnormalize(normed_observations, 'observations')
            else:
                observations = normed_observations
                actions = normed_actions
            trajectories = Trajectories(actions, observations)
            return trajectories
        else:
            normed_observations = trajectories[:, :, self.action_dim:]
            if it:
                observations = self.normalizer.unnormalize(normed_observations, 'observations')
            else:
                observations = normed_observations
            trajectories = Trajectories_obs(observations)
            return trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size, it=1):
        if it:
            conditions = utils.apply_dict(
                self.normalizer.normalize,
                conditions,
                'observations',
            )
            conditions =  {
		        k: v.squeeze()
		        for k, v in conditions.items()
	        }
        else:
            conditions =  {
		        k: v
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
