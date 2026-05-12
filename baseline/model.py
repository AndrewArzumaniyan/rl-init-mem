import torch
import torch.nn as nn

class NPPModel(nn.Module):
    def __init__(self):
        super(NPPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, g_start, delta_g):
        x = torch.cat([g_start, delta_g], dim=1)
        return self.net(x)