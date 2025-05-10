import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class SingleQueryAttention(nn.Module):
    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()

        self.d_model = d_model

        self.q_linear = nn.Linear(d_in, d_model)
        self.k_linear = nn.Linear(d_in, d_model)
        self.v_linear = nn.Linear(d_in, d_model)

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)

        return output

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        output = self.attention(query, key, value)

        return output


class Attention(nn.Module):
    def __init__(self, d_x: int, d_model: int, d_z: int, d_out: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.q_linear = nn.Linear(d_x, d_model)
        self.k_linear = nn.Linear(d_z, d_model)
        self.v_linear = nn.Linear(d_z, d_out)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        pass
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)

        return output

    def forward(self, x: torch.Tensor, z: torch.Tensor, mask: torch.Tensor):
        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        output = self.attention(query, key, value)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // num_heads  # dimension of each head
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        d_k: int,
        mask: Optional[torch.Tensor] = None,
        drop: Optional[bool] = None,
    ) -> torch.Tensor:
        # size of q, k, v: [batch_size, num_heads, seq_len, d_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if drop is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class GroupQueryAttention(nn.Module):
    pass
