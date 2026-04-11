import torch
import torch.nn as nn

class BranchPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        # the input dim's gonna be # features which we'll know
        # hidden dims can also be expanded as needed
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)