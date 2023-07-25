import torch.nn as nn
import torch

class EBMDiffusionModel(nn.Module):

    def __init__(self, net, condition_type='normal'):
        super().__init__()
        self.net = net
        self.condition_type = condition_type

    def point_energy(self, x, cond, t, fake=False):
        score = self.net(x, cond, t)
        assert score.shape == x.shape
        # if is_pair(x):
        #     score = pair_consistency(score)
        # point_energy = ((score - x) ** 2).sum(-1)
        point_energy = torch.mul(x, score).sum(-1)
        if fake:
            point_energy *= -0.5
        return point_energy
  
    def __call__(self, x, cond, t,fake):
        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.point_energy(x,cond,t,fake).sum(-1).sum()
            gradient = torch.autograd.grad([energy], [x],create_graph=True)[0]
        return gradient
    
    def sample(self, x, cond, t):
        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.point_energy(x,cond,t).sum(-1).sum()
            gradient = torch.autograd.grad([energy], [x])[0]
        return gradient
    
    def get_buffer_energy(self, obs, device): 
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        t = torch.zeros(obs.shape[0], device=device, dtype=torch.float32)
        cond = None
        energy = self.point_energy(obs, cond, t).mean(-1)
        return energy

    def get_target_energy(self, target, device):
        target = torch.tensor(target, device=device, dtype=torch.float32)
        t = torch.zeros(target.shape[0], device=device, dtype=torch.float32)
        cond = None
        if self.condition_type == 'extend':
            target = torch.cat((target, target), dim=2)
        energy = self.point_energy(target, cond, t).mean(-1)
        return energy.detach().cpu().numpy()