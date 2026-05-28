# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import gc
import inspect
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed, use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if use_deterministic:
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def cleanup_memory(verbose=False) -> None:
    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    del_vars = [k for k in list(globals().keys()) if k.startswith("_tmp_")]
    for k in del_vars:
        globals().pop(k, None)
    gc.collect()

    if torch.cuda.is_available():
        # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/33
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        memory_after = total_reserved_mem()
        if verbose:
            print(f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GiB"
                  f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GiB)")


def find_layers(module, layers=None, name=''):
    """
    Recursively finds all instances of specified layers in a module.
    """
    if layers is None:
        layers = [nn.Linear]

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_layers_to_factorize(model_type: str) -> list[str]:
    """Returns a list of sublayers to factorize based on the model architecture."""
    if model_type in ["llama", "mistral", "mixtral", "mobilellm", "qwen3"] or model_type.startswith("gemma"):
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'self_attn.k_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]
    elif model_type == "opt":
        ret = [
            'self_attn.q_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'self_attn.k_proj',
            'fc1',
            'fc2',
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ret


def get_decoder_layers(model):
    """
    Returns the list of decoder layers based on the model architecture.
    """
    model_type = model.config.model_type
    if model_type in ["llama", "mistral", "mixtral", "mobilellm", "qwen3"] or model_type.startswith("gemma"):
        return model.model.layers
    elif model_type == "opt":
        return model.model.decoder.layers
    elif model_type == "gpt2":
        return model.transformer.h
    raise AttributeError(f"Could not find decoder layers for model architecture '{model_type}'.")


def get_decoder_layer_cls_name(model: nn.Module) -> List[str]:
    """Helper to get the class name of the decoder blocks (to prevent accelerate from splitting blocks)."""
    try:
        layers = get_decoder_layers(model)
        if layers:
            return [layers[0].__class__.__name__]
    except AttributeError:
        pass
    return []


def calculate_ranks(model, layers_to_analyze, quant_config):
    """
    Unified entry point for bit allocation.
    """
    def _get_rank(a, b, bits, num_scales=2):
        """
        Estimates split_dim based on bit target, accounting for scale overhead.

        Standard (2 scales: pre, post):
            Total Bits = Paths * [ Rank * (a + b) + 16 * (a + b) ]
        DBF (3 scales: pre, mid, post):
            Total Bits = Paths * [ Rank * (a + b) + 16 * (a + b + Rank) ]
        """
        if bits is None or a * b == 0:
            return None

        total_budget_bits = a * b * bits
        param_sum = a + b

        if num_scales == 3:
            # Rank = (Budget - 16*(a+b)) / (a+b+16)
            return (total_budget_bits - 16 * param_sum) / (param_sum + 16)
        else:
            # Standard: Rank = (Budget / (a+b)) - 16
            return (total_budget_bits / param_sum) - 16

    def _finalize_rank(rank, min_rank):
        curr_rank = int(rank) if rank is not None else 0
        curr_rank = (curr_rank // 32) * 32
        if curr_rank == 0:
            curr_rank = min_rank
        return max(curr_rank, min_rank)

    def _validate_rank(rank, in_features, out_features):
        """Validate that rank is reasonable for layer dimensions."""
        if rank <= 0:
            return max(min(in_features, out_features) // 32, 32)
        if rank > min(in_features, out_features):
            return min(in_features, out_features)
        return rank

    num_scales = 3 if quant_config['admm_type'] == 'dbf' else 2

    print(f"Rank calculation: Bits = ({quant_config['bits']:.2f}), Scales: {num_scales}")
    ranks = {}
    for i, layer in enumerate(get_decoder_layers(model)):
        subset = find_layers(layer)
        for name in layers_to_analyze:
            if name in subset:
                lx = subset[name]
                rank = _get_rank(lx.in_features, lx.out_features, quant_config['bits'], num_scales)
                final_rank = _finalize_rank(rank, 32)
                final_rank = _validate_rank(final_rank, lx.in_features, lx.out_features)
                ranks[f"{i}.{name}"] = final_rank
    return ranks
