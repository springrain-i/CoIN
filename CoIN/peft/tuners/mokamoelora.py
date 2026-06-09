# -*- encoding: utf-8 -*-
# MoE-MoKA: Mixture-of-Experts Multimodal Low-Rank Adaptation
#
# Each expert is a complete MoKA unit: (A_text_i, A_vis_i, B_i) with an
# intra-expert cross-attention. Soft routing weights each expert's full output.
#
# Design (following CoIN MoE-LoRA rank-split convention):
#   r_per = r // expert_num  (each expert gets equal slice of rank budget)
#
#   expert_i forward:
#     a_text = A_text_i · x_text                      (n_text, r_per)
#     a_vis  = A_vis_i  · x_vis                       (n_vis,  r_per)
#     a_vis += CrossAttn(query=a_vis, key=a_text, val=a_text)
#     a_combined[text] = a_text,  a_combined[vis] = a_vis
#     out_i  = B_i · a_combined                       (B, S, d_out)
#
#   final: ΔW·x = Σᵢ wᵢ · out_i · scaling
#
# Token mask convention (CoIN): 2=text, 1=visual, 0=pad.

import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Embedding,
    Conv2d,
)
from ..import_utils import is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MoEMoKALoraConfig(LoraConfig):
    """
    Configuration for MoE-MoKA LoRA.

    Each expert is a full MoKA unit (A_text, A_vis, B) with intra-expert
    cross-attention. rank r is split evenly across expert_num experts, so
    r must be divisible by expert_num.
    """
    expert_num: int = field(
        default=8,
        metadata={"help": "Number of MoE experts. r must be divisible by expert_num."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.MOE_MOKA_CoIN


# ---------------------------------------------------------------------------
# Per-expert MoKA unit
# ---------------------------------------------------------------------------

class MoEMoKAExpert(nn.Module):
    """
    One MoKA expert: independent A_text, A_vis, and B.
    All three operate on per-expert rank r_per = r // expert_num.

    Attribute names start with 'lora_' so mark_only_lora_as_trainable keeps
    them unfrozen.
    """

    def __init__(self, in_features: int, out_features: int, r_per: int):
        super().__init__()
        self.lora_A_text = nn.Linear(in_features, r_per, bias=False)
        self.lora_A_vis  = nn.Linear(in_features, r_per, bias=False)
        self.lora_B      = nn.Linear(r_per, out_features, bias=False)


# ---------------------------------------------------------------------------
# LoRA layer mixin
# ---------------------------------------------------------------------------

class MoEMoKALoraLayer(LoraLayer):
    """
    LoRA layer mixin for MoE-MoKA.

    Stores per-adapter expert lists and the soft router.  Inherits token_mask
    and lora_mode infrastructure from LoraLayer.
    """

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        # adapter_name → ModuleList[MoEMoKAExpert]
        self.lora_experts = nn.ModuleDict({})
        # adapter_name → nn.Linear(in_features, expert_num)  (soft router)
        self.lora_router  = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool,
    ):
        if r % self.expert_num != 0:
            raise ValueError(
                f"MoE-MoKA requires r ({r}) to be divisible by expert_num ({self.expert_num})."
            )
        r_per = r // self.expert_num

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        drop = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: drop}))

        experts = nn.ModuleList(
            [MoEMoKAExpert(self.in_features, self.out_features, r_per)
             for _ in range(self.expert_num)]
        )
        self.lora_experts.update(nn.ModuleDict({adapter_name: experts}))
        self.lora_router.update(
            nn.ModuleDict(
                {adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}
            )
        )

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name: str):
        if adapter_name not in self.lora_experts:
            return
        for expert in self.lora_experts[adapter_name]:
            nn.init.kaiming_uniform_(expert.lora_A_text.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(expert.lora_A_vis.weight,  a=math.sqrt(5))
            nn.init.zeros_(expert.lora_B.weight)          # ΔW=0 at init
        nn.init.normal_(self.lora_router[adapter_name].weight, std=0.01)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class MoEMoKALoraLinear(nn.Linear, MoEMoKALoraLayer):
    """
    Drop-in replacement for nn.Linear with MoE-MoKA adaptation.

    Forward overview:
      1. Base linear:  result  = W₀ · x
      2. Router:       w       = softmax(W_r · x)        (B, S, N)
      3. Per expert i:
           a_text  = A_text_i(x_text)                   (n_text, r_per)
           a_vis   = A_vis_i(x_vis)                     (n_vis,  r_per)
           a_vis  += CrossAttn(a_vis, a_text, a_text)   intra-expert
           a_comb  = [a_text at text positions ;
                      a_vis  at vis  positions]          (B, S, r_per)
           out_i   = B_i · a_comb                       (B, S, d_out)
      4. Aggregate: result += Σᵢ wᵢ · out_i · scaling
    """

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        expert_num: int = 4,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MoEMoKALoraLayer.__init__(self, in_features, out_features, expert_num)

        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        warnings.warn("MoE-MoKA does not support weight merging.")

    def unmerge(self):
        pass

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), self.bias)

        adapter = self.active_adapter
        if (
            self.disable_adapters
            or adapter not in self.lora_experts
            or self.r.get(adapter, 0) <= 0
        ):
            return result.to(previous_dtype)

        scaling = self.scaling[adapter]
        x = x.to(self.lora_experts[adapter][0].lora_A_text.weight.dtype)
        B, S, d_in = x.shape

        # --- Resolve modality masks -------------------------------------------
        token_mask = getattr(self, "token_mask", None)
        lora_mode  = getattr(self, "lora_mode", "all")

        if token_mask is not None:
            if token_mask.shape[1] != S:
                token_mask = token_mask[:, -S:] if S <= token_mask.shape[1] else None
            if token_mask is not None:
                text_mask = (token_mask == 2)   # (B, S)
                vis_mask  = (token_mask == 1)
            else:
                text_mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
                vis_mask  = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        else:
            text_mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
            vis_mask  = torch.zeros(B, S, dtype=torch.bool, device=x.device)

        if lora_mode == "text":
            vis_mask  = torch.zeros_like(vis_mask)
        elif lora_mode == "vision":
            text_mask = torch.zeros_like(text_mask)

        has_text = bool(text_mask.any())
        has_vis  = bool(vis_mask.any())

        # --- Router: per-token soft weights (B, S, N) -------------------------
        # Router uses clean x for stable routing decisions; dropout is applied
        # after routing to regularize only the expert computation path.
        router_weights = F.softmax(
            self.lora_router[adapter](x), dim=-1
        )   # (B, S, N)

        # --- Vectorized per-expert contribution ----------------------------------
        dropped_x = self.lora_dropout[adapter](x)
        flat_text = dropped_x[text_mask]   # (n_text, d_in)
        flat_vis  = dropped_x[vis_mask]    # (n_vis,  d_in)

        expert_sum = self._lora_vectorized(
            flat_text, flat_vis, dropped_x, router_weights,
            adapter, text_mask, vis_mask, has_text, has_vis, B, S,
        )

        result = result + expert_sum * scaling
        return result.to(previous_dtype)

    def _lora_vectorized(
        self,
        flat_text: torch.Tensor,   # (n_text, d_in)
        flat_vis:  torch.Tensor,   # (n_vis,  d_in)
        dropped_x: torch.Tensor,   # (B, S, d_in)  — for shape/device ref
        router_weights: torch.Tensor,  # (B, S, N)
        adapter: str,
        text_mask: torch.Tensor,   # (B, S) bool
        vis_mask:  torch.Tensor,   # (B, S) bool
        has_text: bool,
        has_vis:  bool,
        B: int,
        S: int,
    ) -> torch.Tensor:
        """All N experts computed in parallel via stacked einsum.

        Expert loop O(N) → 2 einsum calls.
        Cross-attention batch loop retained (variable per-sample token counts);
        but now processes all N experts simultaneously per sample.
        """
        experts = self.lora_experts[adapter]
        N = len(experts)

        # Stack weights: (N, r_per, d_in) and (N, d_out, r_per)
        A_text = torch.stack([e.lora_A_text.weight for e in experts], dim=0)
        A_vis  = torch.stack([e.lora_A_vis.weight  for e in experts], dim=0)
        B_all  = torch.stack([e.lora_B.weight       for e in experts], dim=0)

        r_per = A_text.shape[1]
        dev   = dropped_x.device
        dt    = dropped_x.dtype

        # (n_text, N, r_per) and (n_vis, N, r_per) — all experts at once
        out_text_N = (torch.einsum('td,nrd->tnr', flat_text, A_text)
                      if has_text else flat_text.new_zeros(0, N, r_per))
        out_vis_N  = (torch.einsum('vd,nrd->vnr', flat_vis,  A_vis)
                      if has_vis  else flat_vis.new_zeros(0, N, r_per))

        # Cross-attention: batch loop, all N experts simultaneously
        if has_vis and has_text:
            out_vis_N = self._cross_attention_vec(
                out_vis_N, out_text_N, text_mask, vis_mask, B
            )

        # Scatter into (B, S, N, r_per)
        a_comb = torch.zeros(B, S, N, r_per, device=dev, dtype=dt)
        if has_text:
            a_comb[text_mask] = out_text_N
        if has_vis:
            a_comb[vis_mask]  = out_vis_N

        # (B, S, N, r_per) × (N, d_out, r_per) → (B, S, N, d_out)
        out_all = torch.einsum('bsnr,nor->bsno', a_comb, B_all)

        # Router-weighted sum → (B, S, d_out)
        return (router_weights.unsqueeze(-1) * out_all).sum(dim=2)

    @staticmethod
    def _cross_attention_vec(
        query:   torch.Tensor,     # (n_vis,  N, r_per)
        key_val: torch.Tensor,     # (n_text, N, r_per)
        text_mask: torch.Tensor,   # (B, S) bool
        vis_mask:  torch.Tensor,   # (B, S) bool
        batch_size: int,
    ) -> torch.Tensor:
        """Cross-attention for all N experts simultaneously.

        Batch loop is kept because each sample has variable n_text/n_vis.
        Within each sample, all N experts are processed in one einsum.
        """
        out = query.clone()
        vis_offset  = 0
        text_offset = 0
        d_k = query.shape[-1]
        for b in range(batch_size):
            n_text = int(text_mask[b].sum())
            n_vis  = int(vis_mask[b].sum())
            if n_text == 0 or n_vis == 0:
                vis_offset  += n_vis
                text_offset += n_text
                continue

            q  = query  [vis_offset:  vis_offset  + n_vis  ]  # (n_vis,  N, r_per)
            kv = key_val[text_offset: text_offset + n_text ]  # (n_text, N, r_per)

            # scores: (n_vis, n_text, N) — all experts in one einsum
            scores = torch.einsum('vnr,tnr->vtn', q, kv) / math.sqrt(d_k)
            attn   = F.softmax(scores, dim=1)                 # (n_vis, n_text, N)
            update = torch.einsum('vtn,tnr->vnr', attn, kv)  # (n_vis, N, r_per)

            out[vis_offset: vis_offset + n_vis] = q + update

            vis_offset  += n_vis
            text_offset += n_text
        return out


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MoEMoKALoraModel(LoraModel):
    """
    Replaces target Linear modules with MoEMoKALoraLinear.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_mokamoelora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MoEMoKALoraModel supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none'."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
        }

        if isinstance(target, nn.Linear):
            new_module = MoEMoKALoraLinear(
                adapter_name,
                target.in_features,
                target.out_features,
                bias=bias,
                **kwargs,
            )
        else:
            raise ValueError(
                f"MoE-MoKA currently supports only nn.Linear targets; "
                f"got {type(target)}."
            )

        return new_module

    @staticmethod
    def _prepare_mokamoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            model_type = model_config.get("model_type", "")
            if model_type not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError(
                    f"Please specify `target_modules` in `MoEMoKALoraConfig` "
                    f"for model type '{model_type}'."
                )
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_type
            ]
        return peft_config
