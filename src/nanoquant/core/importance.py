# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from tqdm import tqdm

from ..utils.utils import cleanup_memory

# Gradient Scaling Factor to prevent underflow (Numerical Stability)
GRAD_SCALE_FACTOR = 1e6
# --- Hyperparameters ---
# tracks the 99.9th percentile to filter outliers
PERCENTILE = 0.999


# -----------------------------------------------------------------------------
# Core Logic: Robust Max Calculation
# -----------------------------------------------------------------------------
def _get_robust_batch_tau(norms: torch.Tensor, percentile: float = 0.999) -> torch.Tensor:
    """
    Returns the k-th largest value (robust max) as a 0-dim tensor on the same device.
    Avoids GPU->CPU sync from `.item()` inside hooks.
    """
    n = norms.numel()
    if n == 0:
        return norms.new_zeros(())
    k = max(1, int(n * (1.0 - percentile)))
    return torch.topk(norms.reshape(-1), k).values[-1]


# -----------------------------------------------------------------------------
# Hook Functions: Two-Phase & Online Robust Hooks
# -----------------------------------------------------------------------------
def _phase1_robust_profiling_hook(module, inputs, outputs, layer_name, global_stats, is_forward=True):
    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        key = "i_max"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        key = "o_max"

    norms = torch.norm(x, dim=1, keepdim=True)
    tau = _get_robust_batch_tau(norms, PERCENTILE)

    prev = global_stats[key].get(layer_name, None)
    global_stats[key][layer_name] = tau if prev is None else torch.maximum(prev, tau)


def _fixed_clipping_hook(module, inputs, outputs, layer_name, stats_dict, thresholds, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        out_key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        out_key = "o_norm"

    thresh = thresholds["i" if is_forward else "o"].get(layer_name, 1e9)

    norms = torch.norm(x, dim=1, keepdim=True)
    clip_scales = torch.clamp(thresh / (norms + 1e-8), max=1.0)
    x_clipped = x * clip_scales

    update = x_clipped.square().mean(dim=0)
    if not is_forward:
        update = update / GRAD_SCALE_FACTOR

    if update.device != stats_dev:
        update = update.to(stats_dev)
    stats_dict[out_key][layer_name].add_(update)


def _online_clipping_hook(module, inputs, outputs, layer_name, stats_dict, run_states, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float() * GRAD_SCALE_FACTOR
        key = "o_norm"

    norms = torch.norm(x, dim=1, keepdim=True)
    tau = _get_robust_batch_tau(norms, PERCENTILE)

    state = run_states[layer_name][key]
    gmax = state["global_max"]

    if gmax is None:
        gmax = tau
    elif tau > gmax:
        correction = (tau / (gmax + 1e-8)).square()
        stats_dict[key][layer_name].mul_(correction.to(stats_dev))
        gmax = tau

    state["global_max"] = gmax

    clip_scales = torch.clamp(gmax / (norms + 1e-8), max=1.0)
    update = (x * clip_scales).square().mean(dim=0)

    if not is_forward:
        update /= GRAD_SCALE_FACTOR

    stats_dict[key][layer_name].add_(update.to(stats_dev))


# -----------------------------------------------------------------------------
# Hook Functions: DBF Strategy
# -----------------------------------------------------------------------------
def _dbf_hook(module, inputs, outputs, layer_name, stats_dict, stats_device, is_forward=True):
    stats_dev = torch.device(stats_device)

    if is_forward:
        x = inputs[0].detach().flatten(0, -2).float()
        out_key = "i_norm"
    else:
        x = outputs[0].detach().flatten(0, -2).float()
        out_key = "o_norm"

    update = x.square().mean(dim=0)
    if not is_forward:
        update *= GRAD_SCALE_FACTOR

    if update.device != stats_dev:
        update = update.to(stats_dev)
    stats_dict[out_key][layer_name].add_(update)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher):
    """Common calibration loop used by all strategies."""
    def _to_batch(x):
        # preserves your behavior (you used unsqueeze(0) assuming per-sample tensors)
        return x.unsqueeze(0) if x.dim() == 1 else x

    for batch in tqdm(dataloader):
        batch_dev = _to_batch(batch.to(dev, non_blocking=True))
        hidden = _get_last_hidden_state(model, batch_dev, model_offload)
        loss = _calculate_loss(None if model_offload else hidden, hidden if model_offload else None, model, batch_dev,
                               use_truefisher, model_offload)
        loss.backward()
        model.zero_grad(set_to_none=True)


def _get_last_hidden_state(model, batch, model_offload):
    if model.config.model_type == "opt":
        attention_mask = (batch != model.config.pad_token_id).long()
        return model(input_ids=batch, attention_mask=attention_mask).logits if model_offload else \
               model.model.decoder(input_ids=batch, attention_mask=attention_mask)[0]
    return model(batch).logits if model_offload else model.model(batch)[0]


def _calculate_loss(embs, lm_logits, model, batch, use_truefisher, model_offload):
    if use_truefisher:
        with torch.inference_mode():
            step = 1024
            labels = [
                torch.multinomial(torch.softmax(model.lm_head(embs[:, i:i + step]), dim=-1)[0], 1).reshape(1, -1)
                for i in range(0, embs.shape[1], step)
            ]
            labels = torch.cat(labels, dim=1)
        return linear_cross_entropy(embs, model.lm_head.weight, labels)

    if model_offload:
        return F.cross_entropy(
            lm_logits[:, :-1, :].reshape(-1, lm_logits.size(-1)),
            batch[:, 1:].to(lm_logits.device).reshape(-1),
        )
    return linear_cross_entropy(embs, model.lm_head.weight, batch.to(embs.device), shift=1)


def get_shrunk_stats(raw_stats: dict, shrinkage: float = 0.0) -> dict:
    """
    Creates a new statistics dictionary with covariance shrinkage applied.
    
    Args:
        raw_stats (dict): Raw calibration statistics from `collect_stats()`
        shrinkage (float): Shrinkage strength (0.0 = no shrinkage, 1.0 = full shrinkage)

    Returns:
        dict: A new dictionary containing the shrunk statistics.
    """
    # 1. Create a deep copy to prevent "Double Dipping" or polluting the raw data.
    #    We use .clone() on tensors to allocate new memory (approx. 7MB total, negligible).
    shrunk_stats = {
        'i_norm': {
            k: v.clone()
            for k, v in raw_stats['i_norm'].items()
        },
        'o_norm': {
            k: v.clone()
            for k, v in raw_stats['o_norm'].items()
        },
        'stats_device': raw_stats.get('stats_device', 'cpu')
    }

    # 2. Return early if no shrinkage is needed (just return the copy).
    if shrinkage <= 0.0 or shrinkage >= 1.0:
        return shrunk_stats

    print(f"Applying covariance shrinkage (strength: {shrinkage})...")

    # 3. Apply the shrinkage formula to the copied tensors.
    #    H_new = (1 - shrinkage) * H + shrinkage * mean(H)
    for key in ['i_norm', 'o_norm']:
        for layer_name, tensor in shrunk_stats[key].items():
            if tensor.numel() == 0:
                continue

            mean_val = tensor.mean()
            # In-place modification is safe here because 'tensor' is already a clone.
            tensor.mul_(1.0 - shrinkage).add_(mean_val * shrinkage)

    return shrunk_stats


def register_stats(model, stats: dict):
    """
    Registers the calibration statistics (i_norm, o_norm) as buffers to the model.
    
    Args:
        model (nn.Module): The PyTorch model to modify.
        stats (dict): The dictionary containing processed 'i_norm' and 'o_norm'.

    Returns:
        model: The model with registered buffers.
    """
    # 1. Identify target linear layers (excluding the LM head).
    linear_layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}

    device = stats.get('stats_device', 'cpu')
    count = 0

    print(f"Attaching statistics to {len(linear_layers)} layers...")

    for name, m in linear_layers.items():
        # 2. Register Input Norm (i_norm)
        if name in stats['i_norm']:
            # persistent=False means these buffers won't be saved in the model's state_dict
            # (checkpoints), keeping the file size small.
            m.register_buffer("i_norm", stats['i_norm'][name], persistent=False)
        else:
            # Fallback: If stats are missing, register a tensor of ones (Identity op).
            m.register_buffer("i_norm", torch.ones(m.weight.shape[1], device=device), persistent=False)
        # 3. Register Output Norm (o_norm)
        if name in stats['o_norm']:
            m.register_buffer("o_norm", stats['o_norm'][name], persistent=False)
        else:
            m.register_buffer("o_norm", torch.ones(m.weight.shape[0], device=device), persistent=False)
        count += 1

    print(f"Registered i_norm and o_norm buffers to {len(linear_layers)} layers")
    return model


# -----------------------------------------------------------------------------
# MAIN CALIBRATION FUNCTION
# -----------------------------------------------------------------------------
def collect_stats(model, dataloader, dev, use_truefisher=False, model_offload=False, vram_limit_gb=50, save_plots=False,
                  strategy='online'):
    """
    Main entry point for NanoQuant calibration statistics collection.
    Collects raw calibration statistics without applying shrinkage.
    """
    stats_device = 'cpu' if model_offload else dev
    print(f"Calibration Strategy: {strategy.upper()} | Offload: {model_offload} (Limit: {vram_limit_gb}GB)")

    linear_layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}

    if not model_offload:
        model.to(dev)

    model.train()
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    stats = {
        'i_norm': defaultdict(lambda: torch.zeros(0, dtype=torch.float32, device=stats_device)),
        'o_norm': defaultdict(lambda: torch.zeros(0, dtype=torch.float32, device=stats_device)),
    }
    handles = []

    # =========================================================
    # STRATEGY: TWO-PHASE (Robust Profiling + Fixed Clipping)
    # =========================================================
    if strategy == 'two_phase':
        print(">>> Phase 1: Robust Profiling (Percentile-based Tau discovery)...")

        global_profiling = {"i_max": {}, "o_max": {}}

        for n, m in linear_layers.items():
            handles.append(
                m.register_forward_hook(
                    partial(_phase1_robust_profiling_hook, layer_name=n, global_stats=global_profiling,
                            is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_phase1_robust_profiling_hook, layer_name=n, global_stats=global_profiling,
                            is_forward=False)))

        _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher)

        for h in handles:
            h.remove()
        handles = []

        thresholds = {
            'i': {
                n: v.item()
                for n, v in global_profiling["i_max"].items()
            },
            'o': {
                n: v.item()
                for n, v in global_profiling["o_max"].items()
            },
        }
        del global_profiling

        print(">>> Phase 2: Sanitized Calibration...")

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], device=stats_device)
            handles.append(
                m.register_forward_hook(
                    partial(_fixed_clipping_hook, layer_name=n, stats_dict=stats, thresholds=thresholds,
                            stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_fixed_clipping_hook, layer_name=n, stats_dict=stats, thresholds=thresholds,
                            stats_device=stats_device, is_forward=False)))

    # =========================================================
    # STRATEGY: ONLINE (Cumulative Monotonic Update)
    # =========================================================
    elif strategy == 'online':
        print(">>> Single Pass: Online Cumulative Preconditioning...")

        run_states = defaultdict(lambda: {'i_norm': {'global_max': None}, 'o_norm': {'global_max': None}})

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], device=stats_device)
            handles.append(
                m.register_forward_hook(
                    partial(_online_clipping_hook, layer_name=n, stats_dict=stats, run_states=run_states,
                            stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_online_clipping_hook, layer_name=n, stats_dict=stats, run_states=run_states,
                            stats_device=stats_device, is_forward=False)))

    # =========================================================
    # STRATEGY: DBF (Direct Buffer Accumulation)
    # =========================================================
    elif strategy == 'dbf':
        print(">>> DBF Strategy: Direct Accumulation...")

        for n, m in linear_layers.items():
            stats['i_norm'][n] = torch.zeros(m.weight.shape[1], dtype=torch.float32, device=stats_device)
            stats['o_norm'][n] = torch.zeros(m.weight.shape[0], dtype=torch.float32, device=stats_device)

            handles.append(
                m.register_forward_hook(
                    partial(_dbf_hook, layer_name=n, stats_dict=stats, stats_device=stats_device, is_forward=True)))
            handles.append(
                m.register_full_backward_hook(
                    partial(_dbf_hook, layer_name=n, stats_dict=stats, stats_device=stats_device, is_forward=False)))

    else:
        for h in handles:
            h.remove()
        raise ValueError(f"Unknown strategy: {strategy}")

    # =========================================================
    # Common calibration loop for all strategies
    # =========================================================
    _run_calibration_loop(dataloader, model, dev, model_offload, use_truefisher)

    # =========================================================
    # Finalization
    # =========================================================
    for h in handles:
        h.remove()

    if torch.cuda.is_available():
        cleanup_memory()

    print("Finalizing raw statistics...")

    n_samples = len(dataloader)

    # Create a copy of raw statistics to return
    raw_stats = {'i_norm': {}, 'o_norm': {}, 'n_samples': n_samples, 'stats_device': stats_device}

    for name, m in linear_layers.items():
        if name in stats['i_norm']:
            raw_stats['i_norm'][name] = stats['i_norm'][name]
            if strategy != "dbf":
                raw_stats['i_norm'][name] /= n_samples
        else:
            raw_stats['i_norm'][name] = torch.ones(m.weight.shape[1], device=stats_device)

        if name in stats['o_norm']:
            raw_stats['o_norm'][name] = stats['o_norm'][name]
            if strategy != "dbf":
                raw_stats['o_norm'][name] /= n_samples
        else:
            raw_stats['o_norm'][name] = torch.ones(m.weight.shape[0], device=stats_device)

    del stats
    return raw_stats
