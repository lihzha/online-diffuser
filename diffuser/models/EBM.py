import torch.nn as nn
import torch

class EBMDiffusionModel(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def point_energy(self, x, cond, t):
        score = self.net(x, cond, t)
        point_energy = ((score - x) ** 2).mean(-1)
        return point_energy
  
    def __call__(self, x, cond, t):
        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.point_energy(x,cond,t).sum()
            gradient = torch.autograd.grad([energy], [x],create_graph=True)[0]
        return gradient
    
    def get_buffer_energy(self, obs, device): 
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        t = torch.zeros(obs.shape[0], device=device, dtype=torch.float32)
        cond = None
        energy = self.point_energy(obs, cond, t).mean(-1)
        return energy

    def get_target_energy(self, target_pair, device):
        target_pair = torch.tensor(target_pair, device=device, dtype=torch.float32)
        t = torch.zeros(target_pair.shape[0], device=device, dtype=torch.float32)
        cond = None
        energy = self.point_energy(target_pair, cond, t).mean(-1)
        return energy.detach().cpu().numpy()