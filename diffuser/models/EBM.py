import torch.nn as nn
import torch

class EBMDiffusionModel(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def point_energy(self, x, cond, t):
        score = self.net(x, cond, t)
        point_energy = ((score - x) ** 2).sum(-1)
        return point_energy
  
    def __call__(self, x, cond, t):
        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.point_energy(x,cond,t).sum()
            gradient = torch.autograd.grad([energy], [x],create_graph=True)[0]
        return gradient
    
