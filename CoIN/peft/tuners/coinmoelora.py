# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
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
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class CoINMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CoIN`]
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN


class CoINMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_coinmoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
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
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = CoINMOELoraLinear(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_coinmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class CoINMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: CoINMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: CoINMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

def _moe_lora_compute(
    lora_x: torch.Tensor,   # (B, T, d_in)
    A: torch.Tensor,        # (N, r_per, d_in)
    B: torch.Tensor,        # (N, d_out, r_per)
    router: torch.Tensor,   # (B, T, N)
    scaling: float,
) -> torch.Tensor:
    """Pure einsum MoE-LoRA computation — extracted for torch.compile."""
    out_A = torch.einsum('bti,nri->btnr', lora_x, A)   # (B,T,N,r_per)
    out_B = torch.einsum('btnr,nor->btno', out_A, B)    # (B,T,N,d_out)
    return (out_B * router.unsqueeze(-1)).sum(dim=2) * scaling


import os as _os_module
_USE_COMPILED_LORA = _os_module.environ.get("COIN_USE_COMPILED_LORA", "0") == "1"
_moe_lora_compute_compiled = None
if _USE_COMPILED_LORA:
    try:
        _moe_lora_compute_compiled = torch.compile(
            _moe_lora_compute, fullgraph=False, dynamic=True
        )
    except Exception:
        _moe_lora_compute_compiled = None


class CoINMOELoraLinear(nn.Linear, CoINMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CoINMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num)
        
        # init the Gate network
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        # Controlled by env var COIN_USE_VECTORIZED_LORA (default: enabled)
        import os as _os
        self.use_vectorized_lora = _os.environ.get("COIN_USE_VECTORIZED_LORA", "1") != "0"


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False

    def _lora_vectorized(self, lora_x: torch.Tensor, router: torch.Tensor, active: str) -> torch.Tensor:
        """Vectorized MoE-LoRA: replaces N sequential matmul pairs with 2 batched einsum ops.

        lora_x: (B, T, d_in)
        router:  (B, T, N)   after softmax
        returns: (B, T, d_out)
        """
        A = torch.stack([m.mlp.weight for m in self.lora_A[active].loraA], dim=0)  # (N, r_per, d_in)
        B = torch.stack([m.mlp.weight for m in self.lora_B[active].loraB], dim=0)  # (N, d_out, r_per)

        if getattr(self, '_log_gradients', False):
            # Three gradient metrics recorded per logged step:
            #   W_A  (exact):  G^t_A[n,r,d] = Σ_{b,t∈text} g_out_A[b,t,n,r] * lora_x[b,t,d]
            #                  Accumulated over all batch elements. Shape: (N, r, d_in).
            #   W_B  (proxy):  ||g_out_B * text_mask||_F  (full batch, no x needed)
            #   ΔW   (exact):  G^t_dW = Σ_{b,t∈text} g_lora_out[b,t] ⊗ lora_x[b,t]
            #                  Norm via gram trick: ||G^t_dW||_F² = Σ_b (Kg_b ⊙ Kx_b).sum()
            #                  avoids materializing the (d_out × d_in) matrix.
            #
            # Backward firing order (determined by forward graph):
            #   hook_lora_out → hook_B → hook_A
            # x_ref is freed in hook_A only; hook_lora_out reads it first while alive.
            raw_mask   = self.token_mask
            logger_ref = getattr(self, '_grad_logger', None)
            layer_name = getattr(self, '_grad_layer_name', '')
            step       = logger_ref.step if logger_ref is not None else 0
            log_every  = logger_ref.log_every_n_steps if logger_ref is not None else 1

            # Capture A's shape/dtype/device now — safe if ZeRO-3 frees A later.
            A_shape, A_dtype, A_device = A.shape, A.dtype, A.device

            out_A = torch.einsum('bti,nri->btnr', lora_x, A)  # (B,T,N,r_per)
            out_B = torch.einsum('btnr,nor->btno', out_A, B)   # (B,T,N,d_out)

            # Mutable list; freed in hook_A (last to fire).
            x_ref = [lora_x.detach()]  # (B, T, d_in)

            # ── hook_lora_out: exact ΔW metric via gram matrix trick ──────────────
            # ||G^t_b(ΔW)||_F² = trace(Kg_b ⊙ Kx_b), Kg=(T_t,T_t), Kx=(T_t,T_t).
            # No large matrix materialised. Per-sample squared norms are summed.
            def hook_lora_out(g_lora_out):
                # g_lora_out: (B, T, d_out)
                if logger_ref is None or step % log_every != 0:
                    return
                x_cap = x_ref[0]  # still alive; hook_A frees it later
                with torch.no_grad():
                    g_t_sq, g_v_sq = 0.0, 0.0
                    n_text_total, n_vis_total = 0, 0
                    for b in range(raw_mask.shape[0]):
                        t_idx = (raw_mask[b] == 2)
                        v_idx = (raw_mask[b] == 1)
                        n_t, n_v = int(t_idx.sum()), int(v_idx.sum())
                        n_text_total += n_t
                        n_vis_total  += n_v
                        if n_t > 0:
                            g_t = g_lora_out[b, t_idx]    # (T_t, d_out)
                            x_t = x_cap[b, t_idx]          # (T_t, d_in)
                            Kg  = g_t @ g_t.T              # (T_t, T_t)
                            Kx  = x_t @ x_t.T              # (T_t, T_t)
                            g_t_sq += (Kg * Kx).sum().item()
                        if n_v > 0:
                            g_v = g_lora_out[b, v_idx]
                            x_v = x_cap[b, v_idx]
                            Kg  = g_v @ g_v.T
                            Kx  = x_v @ x_v.T
                            g_v_sq += (Kg * Kx).sum().item()
                    logger_ref._pending_dw[(step, layer_name)] = {
                        'g_tdW':  g_t_sq ** 0.5,
                        'g_vdW':  g_v_sq ** 0.5,
                        'n_text': n_text_total,
                        'n_vis':  n_vis_total,
                    }

            # ── hook_B: proxy metric for W_B (full batch, no per-sample bug) ──────
            def hook_B(g_out_B):
                if logger_ref is None or step % log_every != 0:
                    return
                with torch.no_grad():
                    tm = (raw_mask == 2).float()[:, :, None, None]
                    vm = (raw_mask == 1).float()[:, :, None, None]
                    logger_ref._pending[(step, layer_name)] = {
                        'g_tB': (g_out_B * tm).norm().item(),
                        'g_vB': (g_out_B * vm).norm().item(),
                    }

            # ── hook_A: exact W_A metric, all batch elements ──────────────────────
            # Fires last — accumulates over batch, pops both pending dicts, flushes.
            def hook_A(g_out_A):
                # g_out_A: (B, T, N, r_per)
                x_cap = x_ref[0]
                x_ref[0] = None  # release — all hooks that need x have already fired
                if logger_ref is None or step % log_every != 0:
                    return
                with torch.no_grad():
                    G_t = torch.zeros(A_shape, dtype=A_dtype, device=A_device)
                    G_v = torch.zeros(A_shape, dtype=A_dtype, device=A_device)
                    for b in range(raw_mask.shape[0]):
                        t_idx = (raw_mask[b] == 2)
                        v_idx = (raw_mask[b] == 1)
                        if t_idx.any():
                            G_t += torch.einsum(
                                'tnr,td->nrd', g_out_A[b, t_idx], x_cap[b, t_idx])
                        if v_idx.any():
                            G_v += torch.einsum(
                                'tnr,td->nrd', g_out_A[b, v_idx], x_cap[b, v_idx])

                    entry_B  = logger_ref._pending.pop((step, layer_name), {})
                    entry_dW = logger_ref._pending_dw.pop((step, layer_name), {})
                    logger_ref._flush(
                        step, layer_name,
                        G_t.norm().item(), G_v.norm().item(),
                        entry_B.get('g_tB',  0.0), entry_B.get('g_vB',  0.0),
                        entry_dW.get('g_tdW', 0.0), entry_dW.get('g_vdW', 0.0),
                        entry_dW.get('n_text', 0),  entry_dW.get('n_vis',  0),
                    )

            lora_out = (out_B * router.unsqueeze(-1)).sum(dim=2) * self.scaling[active]

            if lora_out.requires_grad:
                lora_out.register_hook(hook_lora_out)
            if out_B.requires_grad:
                out_B.register_hook(hook_B)
            if out_A.requires_grad:
                out_A.register_hook(hook_A)

            return lora_out

        compute_fn = _moe_lora_compute_compiled if _moe_lora_compute_compiled is not None else _moe_lora_compute
        return compute_fn(lora_x, A, B, router, self.scaling[active])

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            active = self.active_adapter
            x = x.to(self.lora_A[active].loraA[0].weight.dtype)
            lora_x, _ = self._apply_token_mask(x)
            lora_x = self.lora_dropout[active](lora_x)
            self.lora_router = self.lora_router.to(lora_x.device)
            router = self.lora_router[active](lora_x)
            router = torch.softmax(router, dim=-1)

            if getattr(self, 'use_vectorized_lora', True):
                result = result + self._lora_vectorized(lora_x, router, active)
            else:
                for i in range(self.expert_num):
                    result += (
                        self.lora_B[active].loraB[i](
                            self.lora_A[active].loraA[i](lora_x),
                        )
                        * self.scaling[active]
                        * router[:,:,i].unsqueeze(-1)
                    )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
    


class CoINMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(CoINMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class CoINMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(CoINMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs



class CoINMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y



class CoINMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class CoINMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: CoINMOELoraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits