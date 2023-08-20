import torch.nn as nn

class InverseDynamics(nn.Module):

    def __init__(self, inp_dim, hid_dim, out_dim, device):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.mlp.to(device)

    def forward(self, x):

        out = self.mlp(x)
        return out