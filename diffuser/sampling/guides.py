import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class gt_DensityGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        prob = self.model(x)
        return prob

    def gradients(self, x):
        x.requires_grad_()
        prob = self(x[:,:,2:4])
        grad = torch.autograd.grad([prob.sum()], [x])[0]
        x.detach()
        return prob, grad
    
    
class EBM_DensityGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        prob = self.model(x)
        return prob

    def gradients(self, x, cond, t):
        x.requires_grad_()
        prob = self.model.neg_logp_unnorm(x, cond, t)
        guidance = torch.autograd.grad([prob[:,-1].sum()], [x],retain_graph=True)[0]
        grad = torch.autograd.grad([prob.sum()], [x])[0]
        x.detach()
        return grad, guidance