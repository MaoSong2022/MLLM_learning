import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim) * std_dev)

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha * (x @ self.A @ self.B)

        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()

        self.linear = linear
        self.lora_layer = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_layer(x)