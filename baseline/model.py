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
            nn.Linear(32, 1)
        )

    def forward(self, g_start, g_target):
        x = torch.cat([g_start, g_target], dim=1)
        out = self.net(x)
        return out