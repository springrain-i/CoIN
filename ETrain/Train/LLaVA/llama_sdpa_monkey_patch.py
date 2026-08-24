"""
SDPA-based LlamaAttention monkey patch.

Replaces the flash_attn packed-QKV patch with PyTorch's built-in
F.scaled_dot_product_attention, which:
  - works for both training prefill (Q=KV, causal) and
    eval autoregressive decode (Q=1, KV=N)
  - automatically selects the best backend (Flash / MemEfficient / Math)
    on RTX 3090 (SM 8.6) the Flash backend is used for prefill

Usage:
    from ETrain.Train.LLaVA.llama_sdpa_monkey_patch import replace_llama_attn_with_sdpa
    replace_llama_attn_with_sdpa()

Environment toggle (default on):
    COIN_USE_SDPA_PATCH=0   # disable, fall back to original transformers attention
"""

import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def forward_sdpa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "output_attentions is not supported with the SDPA patch; returning None."
        )

    bsz, q_len, _ = hidden_states.size()

    # Generate position_ids when not provided (required for correct RoPE in decode)
    if position_ids is None:
        past_length = past_key_value[0].shape[-2] if past_key_value is not None else 0
        position_ids = torch.arange(
            past_length, past_length + q_len,
            dtype=torch.long, device=hidden_states.device,
        ).unsqueeze(0).expand(bsz, -1)

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ── decide is_causal and attn_mask ──────────────────────────────────────
    #
    # attention_mask here is the 2D key_padding_mask (bsz, kv_seq_len) or
    # None, because _prepare_decoder_attention_mask is overridden to return it
    # as-is (same override kept from the flash_attn patch).
    #
    # Three cases:
    #  A) prefill, no padding  →  is_causal=True,  attn_mask=None  [fastest]
    #  B) prefill, padding     →  is_causal=False, attn_mask = causal ∩ pad
    #  C) decode (q_len < kv)  →  is_causal=False, attn_mask = pad or None

    is_decode = (q_len < kv_seq_len)  # eval: q=1, kv=full sequence

    if not is_decode and attention_mask is None:
        # Case A: most common training path — let SDPA handle causality
        attn_mask = None
        is_causal = True
    else:
        # Cases B & C: build 4D additive mask explicitly
        # Start with zeros (all positions valid)
        attn_mask = torch.zeros(
            bsz, 1, q_len, kv_seq_len,
            dtype=query_states.dtype, device=query_states.device,
        )

        # Apply padding mask if provided (2D → 4D broadcast)
        if attention_mask is not None:
            # attention_mask: 1=keep, 0=mask
            pad_mask = (attention_mask[:, None, None, :] == 0)  # (bsz,1,1,kv)
            attn_mask = attn_mask.masked_fill(pad_mask, float("-inf"))

        # Apply causal mask for prefill (not needed for decode)
        if not is_decode:
            causal = torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=query_states.device)
            causal = torch.tril(causal)  # lower-triangular = valid
            attn_mask = attn_mask.masked_fill(~causal[None, None], float("-inf"))

        is_causal = False  # mask already encodes causality when needed

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    # With left padding, a padding query in the prefill pass has no valid
    # key after combining the causal and key-padding masks.  Its attention
    # row is therefore fully masked.  Do not let a non-finite value from that
    # unused row propagate through residual connections into real tokens.
    if attention_mask is not None and not is_decode:
        query_is_padding = attention_mask[:, :q_len].eq(0)
        attn_output = attn_output.masked_fill(
            query_is_padding[:, None, :, None], 0.0
        )

    attn_output = (
        attn_output.transpose(1, 2)
        .contiguous()
        .view(bsz, q_len, self.num_heads * self.head_dim)
    )
    return self.o_proj(attn_output), None, past_key_value


def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # Return raw 2D key_padding_mask; the SDPA forward builds the full mask.
    return attention_mask


def replace_llama_attn_with_sdpa():
    if os.environ.get("COIN_USE_SDPA_PATCH", "1") == "0":
        return

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_sdpa
