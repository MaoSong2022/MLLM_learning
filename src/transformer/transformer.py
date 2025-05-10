import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from .attention import MultiHeadAttention
from .normalization import LayerNormalization
from .feed_forward import FeedForward


class EncoderTransformerLayer(nn.Module):
    pass


class DecoderTransformerLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention1 = MultiHeadAttention(num_heads, d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = LayerNormalization(d_model)

        self.attention2 = MultiHeadAttention(num_heads, d_model, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = LayerNormalization(d_model)

        self.feed_forward = FeedForward(d_model, d_model, dropout)
        self.layer_norm3 = LayerNormalization(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x2 = self.layer_norm1(x)
        x = x + self.dropout1(self.attention1(x2, x2, x2, target_mask))

        x2 = self.layer_norm2(x)
        x = x + self.dropout2(self.attention2(x2, outputs, outputs, source_mask))

        x2 = self.layer_norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))

        return x


class Transformer(nn.Module):
    pass


