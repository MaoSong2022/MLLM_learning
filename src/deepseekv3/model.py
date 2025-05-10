from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_interface import ALL_ATTENTION_FUNCTIONS
from .activations import ACT2FN
from .rope import ROPE_INIT_FUNCTIONS


@dataclass
class ModelConfig:
    num_latents: int
    num_heads: int
    num_hidden_layers: int
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dimension: int
    qk_rope_head_dimension: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_heads: int
    qk_head_dimension: int
    v_head_dimension: int
    attention_bias: bool
    attn_implementation: str
    first_k_dense_replace: int
    max_position_embeddings: int
    intermediate_size: int
    pad_token_id: int
    vocab_size: int

    hidden_act: str
    rms_norm_eps: float = 1e-6


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.hidden_size,), self.weight, eps=self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_local_heads = config.num_local_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dimension = config.qk_nope_head_dimension
        self.qk_rope_head_dimension = config.qk_rope_head_dimension
        self.qk_head_dimension = config.qk_head_dimension
        self.v_head_dimension = config.v_head_dimension

        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layer_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.qk_head_dimension, bias=False
        )

        self.kv_a_proj = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dimension,
            bias=config.attention_bias,
        )
        self.kv_a_layer_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dimension + self.v_head_dimension),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dimension, config.hidden_size, bias=False
        )

        self.scaling = self.qk_head_dimension ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.size()

        query_shape = (batch_size, sequence_length, -1, self.qk_head_dimension)
        key_shape = (
            batch_size,
            sequence_length,
            -1,
            self.qk_nope_head_dimension + self.v_head_dimension,
        )

        # shape: (batch_size, sequence_length, num_heads, qk_head_dimension)
        q_states = (
            self.q_b_proj(self.q_a_layer_norm(self.q_a_proj(hidden_states)))
            .view(query_shape)
            .transpose(1, 2)
        )
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dimension, self.qk_rope_head_dimension], dim=-1
        )

        compressed_kv = self.kv_a_proj(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.qk_nope_head_dimension, self.v_head_dimension], dim=-1
        )

        # shape: (batch_size, sequence_length, num_heads, qk_nope_head_dimension)
        k_pass = (
            self.kv_b_proj(self.kv_a_layer_norm(k_pass)).view(key_shape).transpose(1, 2)
        )
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dimension, self.v_head_dimension], dim=-1
        )

        k_rot = k_rot.view(batch_size, 1, sequence_length, self.qk_rope_head_dimension)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat([q_pass, q_rot], dim=-1)
        key_states = torch.cat([k_pass, k_rot], dim=-1)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config.attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if (
            self.config.attn_implementation == "flash_attention_2"
            and self.qk_head_dimension != self.v_head_dimension
        ):
            attn_output = attn_output[:, :, :, : self.v_head_dimension]

        attn_output = attn_output.reshape(batch_size, sequence_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    pass

    def __init__(self, config: ModelConfig):
        self.config = config


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MultiHeadLatentAttention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)

        self.input_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Model:
    def __init__(self, config: ModelConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self) -> Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        input_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        cache_position,
        **kwargs,
    ):
        pass
