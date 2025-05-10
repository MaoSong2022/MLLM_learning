import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge: bool,
        rank: int = 16,
        lora_alpha: float = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)

        if rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.lora_alpha / self.rank
            self.linear.weight.requires_grad = False

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0 and self.merge:
            output = self.linear(
                x,
                self.linear.weight + self.lora_b @ self.lora_a * self.scale,
                self.linear.bias,
            )
            output = self.dropout(output)

            return output

        return self.linear(x)
