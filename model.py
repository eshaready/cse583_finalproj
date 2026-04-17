import torch
import torch.nn as nn

class BranchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        # the input dim's gonna be # features which we'll know
        # hidden dims can also be expanded as needed
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.feature_net = nn.Sequential(*layers)

        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.feature_net(x)
        return self.head(x)