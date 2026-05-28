# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import time

import torch
import torch.nn as nn
from ..optimi import AdamW
from .admm_dbf import factorize_admm_dbf
from .admm_nq import factorize_admm_nanoquant
from ..modules.linear import NanoQuantLinear
from ..utils.utils import cleanup_memory, find_layers, set_seed


@torch.jit.script
def fused_weighted_mse(pred, tgt, importance):
    return ((pred.float() - tgt.float()).square() * importance).sum()


def get_param_group_config(target_module, binary_lr=1e-5, scale_lr=1e-5, bias_lr=1e-5):
    """
    Get the parameter group config for the optimizer.
    """
    # create param groups
    groups = {'binary': [], 'scale': [], 'bias': []}
    # collect params
    for module in target_module.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            # get tag
            tag = getattr(param, 'optim_group', None)
            # fallback if no tag (for bias)
            if tag is None:
                if param.ndim == 1 and 'bias' in name:
                    tag = 'bias'
                else:
                    continue
            if tag in groups:
                groups[tag].append(param)
    # collect and return param groups with respective lr
    configs = []
    for key, lr in zip(groups.keys(), [binary_lr, scale_lr, bias_lr]):
        if groups[key]:
            configs.append({'params': groups[key], 'lr': lr})
    return configs


@torch.enable_grad()
def tune_nonfact(block, block_inputs, block_target_outputs, importance, kwargs, quant_config):
    # set random seed
    set_seed(quant_config['seed'])
    # get hyperparams
    device = "cuda"
    numel = block_target_outputs.numel()
    batch_size = quant_config['nonfact_batch_size']
    epochs = quant_config['nonfact_epochs']
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    lr = quant_config['nonfact_lr']
    # collect linear weight parameters
    params = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            module.weight.requires_grad = True
            params.append(module.weight)
    assert len(params) > 0, "No linear layers found in the block"
    # get optimizer and lr_scheduler
    optimizer = AdamW(params, lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * lr)
    # optimization loop
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_loss = torch.zeros(1, device=device)
        for i in range(num_samples):
            idx = data_idx[i].item()
            # get output and loss
            y = block(block_inputs[idx:idx + 1], **kwargs)[0]
            loss = fused_weighted_mse(y, block_target_outputs[idx:idx + 1], importance)
            # backprop
            (loss / batch_size).backward()
            # gradient update
            is_update_step = (i + 1) % batch_size == 0 or (i + 1) == num_samples
            if is_update_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # update epoch loss
            epoch_loss += loss.detach()
        cleanup_memory()
        # log loss
        print(f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {(epoch_loss / numel).item():.4e}")

    for p in params:
        p.requires_grad = False
    block.zero_grad(set_to_none=True)
    del params, optimizer


@torch.no_grad()
def factorize_and_replace(layer, name, rank, quant_config):
    """
    Factorizes and replaces a submodule with a quantized version (NanoQuantLinear).
    """
    set_seed(quant_config['seed'])
    lx_orig = find_layers(layer)[name]
    original_weight = lx_orig.weight.data.clone()
    weight_for_factorization = original_weight.clone()
    new_module = lx_orig
    device = "cuda"

    # --- 1. Iterative Factorization and Module Conversion ---
    W_res = weight_for_factorization.clone()

    admm_time = time.time()
    # Select factorization function based on type
    is_transpose = W_res.shape[0] < W_res.shape[1]
    if quant_config['admm_type'] == 'dbf':
        factor_results = factorize_admm_dbf(W_res.to(device), lx_orig.i_norm.to(device), lx_orig.o_norm.to(device),
                                            mid_rank=rank, iters=quant_config['admm_outer_iters'],
                                            is_transpose=is_transpose)
    elif quant_config['admm_type'] == 'nanoquant':
        factor_results = factorize_admm_nanoquant(
            W_res.to(device), lx_orig.i_norm.to(device), lx_orig.o_norm.to(device), mid_rank=rank,
            outer_iters=quant_config['admm_outer_iters'], inner_iters=quant_config['admm_inner_iters'],
            is_transpose=is_transpose, rho_scheduler=quant_config['admm_penalty_scheduler'],
            print_admm_steps=quant_config['admm_print_steps'])
    admm_time = time.time() - admm_time

    # Assemble final factorization results
    final_factor_results = argparse.Namespace(**factor_results)

    # Replace module class and convert
    do_tuning = quant_config['tune_fact']
    new_module.__class__ = NanoQuantLinear
    new_module.__quant_convert__(do_train=do_tuning, rank=rank, factor_results=final_factor_results)

    # --- 2. Finalization ---
    if not do_tuning and new_module.bias is not None and hasattr(lx_orig, 'bias') and lx_orig.bias is not None:
        new_module.bias.data.copy_(lx_orig.bias.data)

    recon_error_raw = (factor_results["W_final"].cpu() - weight_for_factorization.cpu()).square().sum().item()
    original_norm_sq = weight_for_factorization.square().sum().item()
    per_el_error = recon_error_raw / W_res.numel()
    if original_norm_sq > 0:
        normalized_error = recon_error_raw / original_norm_sq
        print(
            f"\t\tADMM weight recon error: raw={recon_error_raw:.4f}, norm={normalized_error:.4f}, per_el={per_el_error:.4e}, ADMM time={admm_time:.2f}s"
        )

    del original_weight, W_res, lx_orig
    cleanup_memory()

    return new_module, final_factor_results


@torch.enable_grad()
def tune_fact(block, target_linear, block_inputs, block_target_outputs, importance, kwargs, quant_config):
    # set random seed
    set_seed(quant_config['seed'])
    # get hyperparams
    device = "cuda"
    numel = block_target_outputs.numel()
    batch_size = quant_config['fact_batch_size']
    epochs = quant_config['fact_epochs']
    num_samples = quant_config['num_calib_samples']
    total_steps = math.ceil(num_samples / batch_size) * epochs
    binary_lr = quant_config['fact_binary_lr']
    scale_lr = quant_config['fact_scale_lr']
    bias_lr = quant_config['fact_bias_lr']
    # get optimizer and lr_scheduler
    param_config = get_param_group_config(block, binary_lr=binary_lr, scale_lr=scale_lr, bias_lr=bias_lr)
    optimizer = AdamW(param_config, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4 * scale_lr)
    # optimization loop
    for epoch in range(epochs):
        data_idx = torch.randperm(num_samples, device="cpu", dtype=torch.long)
        epoch_loss = torch.zeros(1, device=device)
        for i in range(num_samples):
            idx = data_idx[i].item()
            # get output and loss
            y = block(block_inputs[idx:idx + 1], **kwargs)[0]
            loss = fused_weighted_mse(y, block_target_outputs[idx:idx + 1], importance)
            # backprop
            (loss / batch_size).backward()
            # gradient update
            is_update_step = (i + 1) % batch_size == 0 or (i + 1) == num_samples
            if is_update_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # update epoch loss
            epoch_loss += loss.detach()
        cleanup_memory()
        # log loss
        print(f"\t\t(Epoch {epoch+1:02d}/{epochs:02d}) Block Loss: {(epoch_loss / numel).item():.4e}")
    # harden latent binary weights
    target_linear.finalize()
    del param_config, optimizer
