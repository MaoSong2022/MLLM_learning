import math
import torch
from torch import nn
from torch.autograd import Variable


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len) -> None:
        super().__init__()

        self.d_model = d_model

        # create a constant PE matrix from pos and i
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # make word embedding larger
        x = x * math.sqrt(self.d_model)

        seq_len = x.size(1)

        # add positional encoding to the word embedding
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()

        return x


class RotationalPositionEncoder(nn.Module):
    pass 