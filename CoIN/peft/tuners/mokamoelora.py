# -*- encoding: utf-8 -*-
# MoE-MoKA: Mixture-of-Experts Multimodal Low-Rank Adaptation
#
# Extends MoKA (modality-specific A matrices, shared B, cross-attention) with
# N experts and per-token soft routing. Design:
#
#   ΔW·x = B · Σᵢ wᵢ · [A_text_i·x_text ; A_vis_i·x_vis + CrossAttn(vis, text)]
#
# - N experts, each holding (A_text_i, A_vis_i) pair mapped to rank-r space.
# - One shared B (r → d_out) per adapter, ensuring cross-modal alignment.
# - Per-token softmax router: wᵢ(x) = softmax(W_router · x)
# - Cross-attention at rank-r cost: visual tokens attend to text tokens.
# - Token routing uses CoIN token_mask convention: 2=text, 1=visual, 0=pad.

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

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

    Adds expert_num on top of standard LoRA config. Each expert holds a
    (A_text, A_vis) pair; all experts share one B matrix.
    """
    expert_num: int = field(
        default=4,
        metadata={"help": "Number of MoE experts."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.MOE_MOKA_CoIN


# ---------------------------------------------------------------------------
# Expert and scalar helper
# ---------------------------------------------------------------------------

class _LearnableScalar(nn.Module):
    """Wraps a single learnable scalar.  Name contains 'lora_' so the
    standard mark_only_lora_as_trainable keeps it unfrozen."""

    def __init__(self, init_val: float = 0.1):
        super().__init__()
        self.lora_value = nn.Parameter(torch.tensor(init_val))

    def get(self) -> torch.Tensor:
        return self.lora_value


class MoEMoKAExpert(nn.Module):
    """Single expert: one A matrix per modality (text / visual)."""

    def __init__(self, in_features: int, r: int):
        super().__init__()
        # Attribute names start with 'lora_' to survive the trainable-param filter.
        self.lora_A_text = nn.Linear(in_features, r, bias=False)
        self.lora_A_vis = nn.Linear(in_features, r, bias=False)


# ---------------------------------------------------------------------------
# LoRA layer mixin
# ---------------------------------------------------------------------------

class MoEMoKALoraLayer(LoraLayer):
    """
    LoRA layer mixin for MoE-MoKA.

    Inherits r, scaling, lora_dropout, and token_mask / lora_mode infrastructure
    from LoraLayer. Adds MoE-specific storage for experts, shared B, router,
    and the cross-attention weight.
    """

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        # Adapter-name → ModuleList[MoEMoKAExpert]
        self.lora_experts = nn.ModuleDict({})
        # Adapter-name → nn.Linear(r, out_features)  — shared B
        self.lora_B_shared = nn.ModuleDict({})
        # Adapter-name → nn.Linear(in_features, expert_num)
        self.lora_router = nn.ModuleDict({})
        # Adapter-name → _LearnableScalar  (cross-attention weight)
        self.lora_attn = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool,
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        drop = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: drop}))

        experts = nn.ModuleList(
            [MoEMoKAExpert(self.in_features, r) for _ in range(self.expert_num)]
        )
        self.lora_experts.update(nn.ModuleDict({adapter_name: experts}))
        self.lora_B_shared.update(
            nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)})
        )
        self.lora_router.update(
            nn.ModuleDict(
                {adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}
            )
        )
        self.lora_attn.update(
            nn.ModuleDict({adapter_name: _LearnableScalar(0.1)})
        )

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name: str):
        if adapter_name not in self.lora_experts:
            return
        for expert in self.lora_experts[adapter_name]:
            nn.init.kaiming_uniform_(expert.lora_A_text.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(expert.lora_A_vis.weight, a=math.sqrt(5))
        # B=0 so ΔW=0 at initialisation, identical to standard LoRA.
        nn.init.zeros_(self.lora_B_shared[adapter_name].weight)
        # Small router init → near-uniform distribution at start.
        nn.init.normal_(self.lora_router[adapter_name].weight, std=0.01)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class MoEMoKALoraLinear(nn.Linear, MoEMoKALoraLayer):
    """
    Drop-in replacement for nn.Linear with MoE-MoKA adaptation.

    Forward pass overview:
      1. Base linear: result = W₀x
      2. Router:      w = softmax(W_r · x)          — shape (B, S, N)
      3. Per expert:  a_i[text] = A_text_i(x[text])
                      a_i[vis]  = A_vis_i(x[vis])
                      a_i[vis] += attn_w · CrossAttn(a_i[vis], a_i[text])
      4. Aggregate:   combined  = Σ_i w_i · a_i     — shape (B, S, r)
      5. Output:      result   += B(combined) * scaling
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

    # merge / unmerge are no-ops (cross-modal architecture is not trivially mergeable)
    def merge(self):
        warnings.warn("MoE-MoKA does not support weight merging.")

    def unmerge(self):
        pass

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        # Always compute the base linear result.
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), self.bias)

        adapter = self.active_adapter
        if (
            self.disable_adapters
            or adapter not in self.lora_experts
            or self.r.get(adapter, 0) <= 0
        ):
            return result.to(previous_dtype)

        r = self.r[adapter]
        scaling = self.scaling[adapter]

        # Cast to the training dtype used by the adapter weights.
        x = x.to(self.lora_experts[adapter][0].lora_A_text.weight.dtype)
        B, S, d_in = x.shape

        # --- Resolve modality masks -------------------------------------------
        token_mask = getattr(self, "token_mask", None)
        lora_mode = getattr(self, "lora_mode", "all")

        if token_mask is not None:
            # During autoregressive generation with KV cache, x has shape (B, 1, d)
            # while token_mask retains the full prompt length.  Coerce to match.
            if token_mask.shape[1] != S:
                token_mask = token_mask[:, -S:] if S <= token_mask.shape[1] else None
            if token_mask is not None:
                text_mask = token_mask == 2  # (B, S)
                vis_mask = token_mask == 1
            else:
                text_mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
                vis_mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        else:
            # No mask: treat every non-padding position as text.
            text_mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
            vis_mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)

        # Partial modality eval: disable one modality path.
        if lora_mode == "text":
            vis_mask = torch.zeros_like(vis_mask)
        elif lora_mode == "vision":
            text_mask = torch.zeros_like(text_mask)

        has_text = bool(text_mask.any())
        has_vis = bool(vis_mask.any())

        # --- Router: per-token soft weights over N experts --------------------
        dropped_x = self.lora_dropout[adapter](x)
        # router_weights: (B, S, N)
        router_weights = F.softmax(self.lora_router[adapter](dropped_x), dim=-1)

        # --- Accumulate expert contributions ---------------------------------
        combined = torch.zeros(B, S, r, device=x.device, dtype=x.dtype)
        attn_w = self.lora_attn[adapter].get()

        for i, expert in enumerate(self.lora_experts[adapter]):
            a_out = torch.zeros(B, S, r, device=x.device, dtype=x.dtype)

            # Always call both A matrices so ZeRO-3 traces a consistent module
            # call order every step.  Pass an empty slice when the modality is
            # absent; the result is discarded via the mask assignment below.
            flat_text = dropped_x[text_mask]  # (n_text, d_in)  — may be empty
            flat_vis = dropped_x[vis_mask]    # (n_vis,  d_in)  — may be empty
            out_text = expert.lora_A_text(flat_text)
            out_vis = expert.lora_A_vis(flat_vis)

            if has_text:
                a_out[text_mask] = out_text
            if has_vis:
                a_out[vis_mask] = out_vis

            # Cross-attention: visual tokens (query) attend to text tokens (k/v).
            # Operates in the cheap rank-r space; no extra projection matrices.
            if has_text and has_vis:
                a_out = self._cross_attention(a_out, text_mask, vis_mask, B, attn_w)

            # Weighted accumulation: w_i is (B, S), broadcast over r-dim.
            combined = combined + router_weights[:, :, i].unsqueeze(-1) * a_out

        # --- Shared B ---------------------------------------------------------
        result = result + self.lora_B_shared[adapter](combined) * scaling

        return result.to(previous_dtype)

    def _cross_attention(
        self,
        a_out: torch.Tensor,
        text_mask: torch.Tensor,
        vis_mask: torch.Tensor,
        batch_size: int,
        attn_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each sample: visual token representations (after A_vis) attend to
        text token representations (after A_text) as query→(key, value).
        All computation is in rank-r space, so it is cheap even for long seqs.
        """
        for b in range(batch_size):
            text_idx = torch.where(text_mask[b])[0]
            vis_idx = torch.where(vis_mask[b])[0]
            if len(text_idx) == 0 or len(vis_idx) == 0:
                continue

            query = a_out[b, vis_idx, :].unsqueeze(0)   # (1, n_vis,  r)
            key = a_out[b, text_idx, :].unsqueeze(0)     # (1, n_text, r)
            value = key

            d_k = query.shape[-1]
            score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            attn_probs = F.softmax(score, dim=-1)
            attn_out = torch.matmul(attn_probs, value).squeeze(0)  # (n_vis, r)

            a_out[b, vis_idx, :] = a_out[b, vis_idx, :] + attn_w * attn_out

        return a_out


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MoEMoKALoraModel(LoraModel):
    """
    Replaces target Linear modules with MoEMoKALoraLinear.
    Identical orchestration to CoINMOELoraModel; only _create_new_module differs.
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
                # Already wrapped — update in place (multi-adapter scenario).
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
