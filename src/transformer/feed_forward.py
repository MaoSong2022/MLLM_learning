import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x