import torch
from torch import nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = (
            self.gamma
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.beta
        )

        return norm


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps  # variance epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.gamma * x.to(input_type)
