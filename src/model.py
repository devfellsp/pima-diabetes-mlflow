import torch
from torch import nn


class DiabetesMLP(nn.Module):
    """MLP simples para classificação binária no Pima Indians Diabetes."""

    def __init__(self, input_dim: int = 8, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
