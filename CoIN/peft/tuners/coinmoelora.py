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

        logger_ref = getattr(self, '_grad_logger', None)
        layer_name = getattr(self, '_grad_layer_name', '')
        if (getattr(self, '_log_gradients', False)
                and torch.is_grad_enabled()
                and logger_ref is not None
                and logger_ref.should_log_record(layer_name)):
            # Three gradient metrics recorded per logged step (all EXACT):
            #   W_A:  G^t_A[n,r,d] = Σ_{b,t∈text} g_out_A[b,t,n,r] * lora_x[b,t,d]
            #         shape (N,r,d_in) — materialised, tiny
            #   W_B:  G^t_B[n,d,r] = Σ_{b,t∈text} g_out_B[b,t,n,d] * out_A[b,t,n,r]
            #         shape (N,d_out,r) — materialised, ~0.5MB, freed in hook_B
            #   ΔW:   G^t_ΔW = Σ_{b,t∈text} g_bt^T x_bt  shape (d_out,d_in) ≈33MB bf16
            #         direct construction, exact — acceptable on 8×24GB GPUs
            #
            # Backward order: hook_lora_out → hook_B → hook_A (flushes all)
            # x_ref freed in hook_A; out_A_ref freed in hook_B.
            raw_mask = getattr(self, 'grad_token_mask', None)
            if raw_mask is None:
                raw_mask = self.token_mask
            assistant_query_mask = getattr(self, 'assistant_query_mask', None)
            answer_prefix_query_mask = getattr(self, 'answer_prefix_query_mask', None)
            is_q_proj = layer_name.endswith('q_proj') or '.q_proj' in layer_name
            step = logger_ref.current_step

            A_shape, A_dtype, A_device = A.shape, A.dtype, A.device
            # B shape for G_B zero tensor: (N, d_out, r_per); infer from B itself
            B_shape, B_dtype = B.shape, B.dtype  # (N, d_out, r_per)

            def _stat_tensor(tensor, dtype):
                # Match the MoE-LoRA parameter dtype for observational stats only;
                # hooks do not return gradients and do not alter the training path.
                return tensor if tensor.dtype == dtype else tensor.to(dtype)

            def _align_mask(mask, batch, seq_len, device):
                if mask is None:
                    return None
                if mask.shape[0] != batch:
                    return None
                if mask.shape[1] != seq_len:
                    if seq_len == 1 and mask.shape[1] > 1:
                        mask = mask[:, -1:]
                    else:
                        return None
                return mask.to(device=device)

            log_mask = _align_mask(raw_mask, lora_x.shape[0], lora_x.shape[1], lora_x.device)
            if log_mask is None:
                log_mask = torch.full(
                    (lora_x.shape[0], lora_x.shape[1]),
                    2,
                    dtype=torch.long,
                    device=lora_x.device,
                )
            assistant_query_mask = _align_mask(
                assistant_query_mask, lora_x.shape[0], lora_x.shape[1], lora_x.device)
            answer_prefix_query_mask = _align_mask(
                answer_prefix_query_mask, lora_x.shape[0], lora_x.shape[1], lora_x.device)

            def _zero(shape, dtype):
                return torch.zeros(shape, dtype=dtype, device=A_device)

            def _bucket_store(shape, dtype):
                store = {
                    'vis': _zero(shape, dtype),
                    'prompt': _zero(shape, dtype),
                    'answer': _zero(shape, dtype),
                }
                if is_q_proj:
                    store['assistant_query'] = _zero(shape, dtype)
                    store['answer_prefix_query'] = _zero(shape, dtype)
                else:
                    store['assistant_query'] = None
                    store['answer_prefix_query'] = None
                return store

            def _cosine(a, b):
                if a is None or b is None:
                    return 0.0
                a_flat = a.reshape(-1).float()
                b_flat = b.reshape(-1).float()
                denom = a_flat.norm() * b_flat.norm()
                if denom.item() <= 1e-12:
                    return 0.0
                return float(torch.dot(a_flat, b_flat).div(denom).item())

            def _norm_or_zero(matrix):
                return 0.0 if matrix is None else matrix.norm().item()

            def _summarize(prefix, matrices):
                return {
                    f'G_vis_{prefix}': _norm_or_zero(matrices['vis']),
                    f'G_prompt_{prefix}': _norm_or_zero(matrices['prompt']),
                    f'G_answer_{prefix}': _norm_or_zero(matrices['answer']),
                    f'G_assistant_query_{prefix}': _norm_or_zero(matrices['assistant_query']),
                    f'G_answer_prefix_query_{prefix}': _norm_or_zero(matrices['answer_prefix_query']),
                    f'cos_prompt_vis_{prefix}': _cosine(matrices['prompt'], matrices['vis']),
                    f'cos_answer_vis_{prefix}': _cosine(matrices['answer'], matrices['vis']),
                    f'cos_prompt_answer_{prefix}': _cosine(matrices['prompt'], matrices['answer']),
                    f'cos_assistant_answer_query_{prefix}': _cosine(
                        matrices['assistant_query'], matrices['answer_prefix_query']),
                }

            def _bucket_indices(batch_index):
                mask_b = log_mask[batch_index]
                vis_idx = mask_b == 1
                prompt_idx = mask_b == 2
                answer_idx = mask_b == 3
                if is_q_proj and assistant_query_mask is not None:
                    assistant_idx = assistant_query_mask[batch_index].bool()
                else:
                    assistant_idx = torch.zeros_like(prompt_idx, dtype=torch.bool)
                if is_q_proj and answer_prefix_query_mask is not None:
                    answer_prefix_idx = answer_prefix_query_mask[batch_index].bool()
                else:
                    answer_prefix_idx = torch.zeros_like(prompt_idx, dtype=torch.bool)
                return {
                    'vis': vis_idx,
                    'prompt': prompt_idx,
                    'answer': answer_idx,
                    'assistant_query': assistant_idx,
                    'answer_prefix_query': answer_prefix_idx,
                }

            def _init_expert_rows():
                rows = []
                for expert_id in range(self.expert_num):
                    rows.append({
                        'expert': expert_id,
                        'route_mean_prompt': 0.0,
                        'route_mean_vis': 0.0,
                        'route_mean_answer': 0.0,
                        'route_mass_prompt': 0.0,
                        'route_mass_vis': 0.0,
                        'route_mass_answer': 0.0,
                        'n_prompt': 0,
                        'n_vis': 0,
                        'n_answer': 0,
                    })
                return rows

            def _route_expert_rows():
                rows = _init_expert_rows()
                route_cap = router.detach().to(device=lora_x.device)
                for b in range(log_mask.shape[0]):
                    idx = _bucket_indices(b)
                    for bucket, route_name in (
                        ('prompt', 'prompt'),
                        ('vis', 'vis'),
                        ('answer', 'answer'),
                    ):
                        token_idx = idx[bucket]
                        n_tokens = int(token_idx.sum())
                        if n_tokens == 0:
                            continue
                        route_mass = route_cap[b, token_idx].float().sum(dim=0)
                        for expert_id in range(self.expert_num):
                            rows[expert_id][f'route_mass_{route_name}'] += float(route_mass[expert_id].item())
                            rows[expert_id][f'n_{route_name}'] += n_tokens
                for row in rows:
                    for route_name in ('prompt', 'vis', 'answer'):
                        n_tokens = row[f'n_{route_name}']
                        if n_tokens > 0:
                            row[f'route_mean_{route_name}'] = row[f'route_mass_{route_name}'] / n_tokens
                return rows

            def _summarize_expert_matrices(prefix, matrices):
                rows = []
                for expert_id in range(self.expert_num):
                    prompt = matrices['prompt'][expert_id]
                    vis = matrices['vis'][expert_id]
                    answer = matrices['answer'][expert_id]
                    rows.append({
                        'expert': expert_id,
                        f'expert_grad_{prefix}_prompt': prompt.norm().item(),
                        f'expert_grad_{prefix}_vis': vis.norm().item(),
                        f'expert_grad_{prefix}_answer': answer.norm().item(),
                        f'cos_{prefix}_prompt_vis': _cosine(prompt, vis),
                        f'cos_{prefix}_answer_vis': _cosine(answer, vis),
                        f'cos_{prefix}_prompt_answer': _cosine(prompt, answer),
                    })
                return rows

            def _merge_expert_rows(base_rows, *updates):
                rows = [dict(row) for row in base_rows]
                for update_rows in updates:
                    for update in update_rows:
                        expert_id = int(update.get('expert', -1))
                        if 0 <= expert_id < len(rows):
                            rows[expert_id].update({k: v for k, v in update.items() if k != 'expert'})
                return rows

            def _summarize_expert_dw(g_lora_out, x_cap):
                rows = []
                route_cap = router.detach().to(device=lora_x.device)
                d_out, d_in = B_shape[1], A_shape[2]
                for expert_id in range(self.expert_num):
                    matrices = {
                        'prompt': torch.zeros(d_out, d_in, dtype=A_dtype, device=A_device),
                        'vis': torch.zeros(d_out, d_in, dtype=A_dtype, device=A_device),
                        'answer': torch.zeros(d_out, d_in, dtype=A_dtype, device=A_device),
                    }
                    for b in range(log_mask.shape[0]):
                        idx = _bucket_indices(b)
                        for bucket in ('prompt', 'vis', 'answer'):
                            token_idx = idx[bucket]
                            if token_idx.any():
                                weights = route_cap[b, token_idx, expert_id].to(dtype=A_dtype).unsqueeze(-1)
                                weighted_grad = _stat_tensor(g_lora_out[b, token_idx], A_dtype) * weights
                                matrices[bucket] += weighted_grad.T @ _stat_tensor(x_cap[b, token_idx], A_dtype)
                    prompt = matrices['prompt']
                    vis = matrices['vis']
                    answer = matrices['answer']
                    rows.append({
                        'expert': expert_id,
                        'expert_grad_dW_prompt': prompt.norm().item(),
                        'expert_grad_dW_vis': vis.norm().item(),
                        'expert_grad_dW_answer': answer.norm().item(),
                        'cos_dW_prompt_vis': _cosine(prompt, vis),
                        'cos_dW_answer_vis': _cosine(answer, vis),
                        'cos_dW_prompt_answer': _cosine(prompt, answer),
                    })
                return rows

            out_A = torch.einsum('bti,nri->btnr', lora_x, A)  # (B,T,N,r_per)
            out_B = torch.einsum('btnr,nor->btno', out_A, B)   # (B,T,N,d_out)

            x_ref     = [lora_x.detach()]    # (B,T,d_in)  freed in hook_A
            out_A_ref = [out_A.detach()]     # (B,T,N,r)   freed in hook_B — tiny ~0.36MB

            # ── hook_lora_out: exact ΔW = Σ_{b,t} g_bt^T x_bt ──────────────────
            # G^t shape (d_out, d_in) = (4096, 4096) ≈ 33MB bf16 — acceptable on 8×24GB
            def hook_lora_out(g_lora_out):
                # g_lora_out: (B, T, d_out)
                if logger_ref is None:
                    return
                x_cap = x_ref[0]  # still alive; hook_A frees it later
                d_out, d_in = B_shape[1], A_shape[2]
                with torch.no_grad():
                    matrices = _bucket_store((d_out, d_in), A_dtype)
                    counts = {
                        'n_vis': 0,
                        'n_prompt': 0,
                        'n_answer': 0,
                        'n_assistant_query': 0,
                        'n_answer_prefix_query': 0,
                    }
                    for b in range(log_mask.shape[0]):
                        idx = _bucket_indices(b)
                        counts['n_vis'] += int(idx['vis'].sum())
                        counts['n_prompt'] += int(idx['prompt'].sum())
                        counts['n_answer'] += int(idx['answer'].sum())
                        counts['n_assistant_query'] += int(idx['assistant_query'].sum())
                        counts['n_answer_prefix_query'] += int(idx['answer_prefix_query'].sum())
                        for bucket, token_idx in idx.items():
                            if token_idx.any():
                                matrices[bucket] += (
                                    _stat_tensor(g_lora_out[b, token_idx], A_dtype).T
                                    @ _stat_tensor(x_cap[b, token_idx], A_dtype)
                                )
                    logger_ref._pending_dw[(step, layer_name)] = {
                        'metrics': _summarize('dW', matrices),
                        'counts': counts,
                        'expert_rows': _merge_expert_rows(
                            _route_expert_rows(),
                            _summarize_expert_dw(g_lora_out, x_cap),
                        ),
                    }

            # ── hook_B: exact W_B metric using out_A ──────────────────────────────
            # out_B[b,t,n,o] = Σ_r out_A[b,t,n,r] * B[n,o,r]
            # → ∂L/∂B[n,o,r] = Σ_{b,t} g_out_B[b,t,n,o] * out_A[b,t,n,r]
            # G^t_B[n,d,r] = Σ_{b,t∈text} g_out_B[b,t,n,d] * out_A[b,t,n,r]
            # shape (N,d_out,r_per) = (8,4096,4) ≈ 0.5MB — fine to materialise
            def hook_B(g_out_B):
                # g_out_B: (B,T,N,d_out)
                out_A_cap    = out_A_ref[0]
                out_A_ref[0] = None   # release
                if logger_ref is None:
                    return
                with torch.no_grad():
                    matrices = _bucket_store(B_shape, B_dtype)
                    for b in range(log_mask.shape[0]):
                        idx = _bucket_indices(b)
                        for bucket, token_idx in idx.items():
                            if token_idx.any():
                                matrices[bucket] += torch.einsum(
                                    'tnd,tnr->ndr',
                                    _stat_tensor(g_out_B[b, token_idx], B_dtype),
                                    _stat_tensor(out_A_cap[b, token_idx], B_dtype),
                                )
                    logger_ref._pending[(step, layer_name)] = {
                        'metrics': _summarize('B', matrices),
                        'expert_rows': _summarize_expert_matrices('B', matrices),
                    }

            # ── hook_A: exact W_A metric, all batch elements ──────────────────────
            # Fires last — accumulates over batch, pops both pending dicts, flushes.
            def hook_A(g_out_A):
                # g_out_A: (B, T, N, r_per)
                x_cap = x_ref[0]
                x_ref[0] = None  # release — all hooks that need x have already fired
                if logger_ref is None:
                    return
                with torch.no_grad():
                    matrices = _bucket_store(A_shape, A_dtype)
                    for b in range(log_mask.shape[0]):
                        idx = _bucket_indices(b)
                        for bucket, token_idx in idx.items():
                            if token_idx.any():
                                matrices[bucket] += torch.einsum(
                                    'tnr,td->nrd',
                                    _stat_tensor(g_out_A[b, token_idx], A_dtype),
                                    _stat_tensor(x_cap[b, token_idx], A_dtype),
                                )

                    entry_B  = logger_ref._pending.pop((step, layer_name), {})
                    entry_dW = logger_ref._pending_dw.pop((step, layer_name), {})
                    metrics = {}
                    metrics.update(_summarize('A', matrices))
                    metrics.update(entry_B.get('metrics', {}))
                    metrics.update(entry_dW.get('metrics', {}))
                    expert_rows = _merge_expert_rows(
                        entry_dW.get('expert_rows', _init_expert_rows()),
                        entry_B.get('expert_rows', []),
                        _summarize_expert_matrices('A', matrices),
                    )
                    logger_ref._flush(
                        step, layer_name,
                        metrics,
                        entry_dW.get('counts', {}),
                    )
                    logger_ref._flush_expert(step, layer_name, expert_rows)

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
            # Router uses clean lora_x (before dropout) for stable, semantically
            # consistent routing decisions. Dropout is applied after routing,
            # regularizing only the expert computation path.
            self.lora_router = self.lora_router.to(lora_x.device)
            router = self.lora_router[active](lora_x)
            router = torch.softmax(router, dim=-1)
            lora_x = self.lora_dropout[active](lora_x)

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
