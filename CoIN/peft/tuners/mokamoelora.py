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
        dropped_x = self.lora_dropout[adapter](x)
        router_weights = F.softmax(
            self.lora_router[adapter](dropped_x), dim=-1
        )   # (B, S, N)

        # --- Per-expert contribution ------------------------------------------
        # Accumulate weighted expert outputs directly in d_out space.
        expert_sum = torch.zeros(B, S, self.out_features, device=x.device, dtype=x.dtype)

        flat_text = dropped_x[text_mask]   # (n_text, d_in)  may be empty
        flat_vis  = dropped_x[vis_mask]    # (n_vis,  d_in)  may be empty

        for i, expert in enumerate(self.lora_experts[adapter]):
            r_per = expert.lora_A_text.out_features

            # Always call both A matrices for consistent ZeRO-3 trace.
            out_text = expert.lora_A_text(flat_text)   # (n_text, r_per)
            out_vis  = expert.lora_A_vis(flat_vis)     # (n_vis,  r_per)

            # Reconstruct full sequence in rank-r_per space.
            a_comb = torch.zeros(B, S, r_per, device=x.device, dtype=x.dtype)
            if has_text:
                a_comb[text_mask] = out_text
            if has_vis:
                # Intra-expert cross-attention before writing back.
                if has_text:
                    out_vis = self._cross_attention(
                        out_vis, out_text, text_mask, vis_mask, B
                    )
                a_comb[vis_mask] = out_vis

            # Per-expert B projects r_per → d_out.
            out_i = expert.lora_B(a_comb)   # (B, S, d_out)

            # Weighted sum: router_weights[:, :, i] is (B, S).
            expert_sum = expert_sum + router_weights[:, :, i].unsqueeze(-1) * out_i

        result = result + expert_sum * scaling
        return result.to(previous_dtype)

    @staticmethod
    def _cross_attention(
        query: torch.Tensor,       # (n_vis,  r_per)  — visual tokens after A_vis
        key_val: torch.Tensor,     # (n_text, r_per)  — text   tokens after A_text
        text_mask: torch.Tensor,   # (B, S) bool
        vis_mask: torch.Tensor,    # (B, S) bool
        batch_size: int,
    ) -> torch.Tensor:
        """
        Intra-expert cross-attention: visual tokens attend to text tokens.
        Operates in rank-r_per space; no extra projection matrices.
        Returns updated visual representations with the same shape as query.
        """
        # We need per-sample indices because each sample may have a different
        # number of text / visual tokens.
        out = query.clone()
        vis_offset = 0
        text_offset = 0
        for b in range(batch_size):
            n_text = int(text_mask[b].sum())
            n_vis  = int(vis_mask[b].sum())
            if n_text == 0 or n_vis == 0:
                vis_offset  += n_vis
                text_offset += n_text
                continue

            q = query  [vis_offset:  vis_offset  + n_vis ].unsqueeze(0)  # (1, n_vis,  r_per)
            k = key_val[text_offset: text_offset + n_text].unsqueeze(0)  # (1, n_text, r_per)
            v = k

            d_k = q.shape[-1]
            score     = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn_probs = F.softmax(score, dim=-1)
            attn_out  = torch.matmul(attn_probs, v).squeeze(0)            # (n_vis, r_per)

            out[vis_offset: vis_offset + n_vis] = query[vis_offset: vis_offset + n_vis] + attn_out

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
