import torch
import torch.nn as nn
from utils import mlp

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_space, hidden_dims=[400, 300], output_activation=nn.Tanh):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.fcs = mlp(num_states, hidden_dims, num_actions, output_activation=output_activation)

    def _normalize(self, action) -> torch.Tensor:
        """
        Normalize the action value to the action space range.
        Hint: the return values of self.fcs is between -1 and 1 since we use tanh as output activation, while we want the action ranges to be (self.action_space.low, self.action_space.high). You can normalize the action value to the action space range linearly.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #

        return (self.action_space.high - self.action_space.low) / 2 * action + \
            (self.action_space.high + self.action_space.low) / 2
        ############################

    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        return super().to(device)

    def forward(self, x):
        # use tanh as output activation
        return self._normalize(self.fcs(x))


class SoftActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min=-20, log_std_max=2):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        """
        Obtain mean and log(std) from fully-connected network.
        Limit the value of log_std to the specified range.
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        param = self.fcs(state)
        if len(param.shape) == 2:
            mean = param[:,:param.shape[-1]//2]
            log_std = torch.clip(param[:,param.shape[-1]//2:], self.log_std_min, self.log_std_max)
        else:
            mean = param[:param.shape[-1]//2]
            log_std = torch.clip(param[param.shape[-1]//2:], self.log_std_min, self.log_std_max)
        ############################
        return mean, log_std

    def evaluate(self, state, sample=True):
        mean, log_std = self.forward(state)
        if not sample:
            return self._normalize(torch.tanh(mean)), None
        # sample action from N(mean, std) if sample is True
        # obtain log_prob for policy and Q function update
        # Hint: remember the reparameterization trick, and perform tanh normalization
        # This library might be helpful: torch.distributions
        ############################
        # YOUR IMPLEMENTATION HERE #
        std = log_std.exp()
        normal = torch.distributions.Normal(0, 1)
        torch.float64
        z = normal.sample().to(std.device)  # unsquashed action
        action = torch.tanh(mean + std*z)
        log_prob1 = torch.distributions.Normal(mean, std).log_prob(mean+std*z) # log_prob1 = -torch.log(std*math.sqrt(2*torch.pi))-z**2/2
        log_prob = log_prob1 - torch.log(1-action**2+1e-6)

        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(-1)
        #     if log_prob[log_prob.argmax()].item()>5:
        #         print("------------------------")
        #         print("log_prob:",log_prob[log_prob.argmax()].item(),"action_1:",1-action[log_prob.argmax()][0].item()**2,"action_2:",1-action[log_prob.argmax()][1].item()**2,"log_tanh:",torch.log(1-action**2+1e-6).sum(-1)[log_prob.argmax()].item(),"log_gaussian:",log_prob1.sum(-1)[log_prob.argmax()].item(),"u_1",(mean+std*z)[log_prob.argmax()][0].item(),"u_2",(mean+std*z)[log_prob.argmax()][1].item())
        # if len(log_prob.shape) > 1:
        #     # log_prob1 = (1-action[:,0]**2).mean().squeeze()
        #     # log_prob2 = (1-action[:,1]**2).mean().squeeze()
        #     log_prob1 = mean.mean(-1).mean()
        #     log_prob2 = std.mean(-1).mean()

        # else:
        #     # log_prob1, log_prob2 = (1-action[0]**2).mean().squeeze(), (1-action[1]**2).mean().squeeze()
        #     log_prob1 = mean.mean(-1).mean()
        #     log_prob2 = std.mean(-1).mean()

        ############################
        return self._normalize(action), log_prob
        


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims):
        super().__init__()
        self.fcs = mlp(num_states + num_actions, hidden_dims, 1)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()
