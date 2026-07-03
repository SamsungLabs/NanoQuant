#!/usr/bin/env python3
# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# This script is based off of the generation script in https://github.com/chu-tianxiang/QuIP-for-all

import argparse
import csv
import gc
import inspect
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM

# add path to include custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # noqa: E402

from ..modules.linear import NanoQuantLinear
from ..utils.load_utils import load_compressed_model  # noqa: F401
from ..utils.load_utils import load_tokenizer
from ..utils.utils import set_seed

torch.set_grad_enabled(False)


def str2bool(value):
    """Convert string to boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected: {value}')


# Optional: energy measurement via Zeus (pip install zeus-ml)
try:
    from zeus.monitor import ZeusMonitor  # type: ignore
except Exception:
    ZeusMonitor = None


# ------------------------------------------------------------
# List parsing
# ------------------------------------------------------------
def _parse_int_list(x: Union[str, int, List[int]], name: str) -> List[int]:
    if isinstance(x, list):
        vals = [int(v) for v in x]
    elif isinstance(x, int):
        vals = [int(x)]
    else:
        s = str(x).strip()
        if s == "":
            raise ValueError(f"--{name} cannot be empty")
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        vals = [int(p) for p in parts] if parts else []
    if not vals:
        raise ValueError(f"--{name} produced an empty list")
    if any(v < 0 for v in vals):
        raise ValueError(f"--{name} must be >= 0")
    return vals


def _tag_from_list(vals: List[int], label: str) -> str:
    if not vals:
        return ""
    uniq = sorted(set(int(v) for v in vals))
    if len(uniq) == 1:
        return f"_{label}-{uniq[0]}"
    return f"_{label}-{uniq[0]}to{uniq[-1]}_n{len(uniq)}"


# ------------------------------------------------------------
# Helpers: filenames / sanitization
# ------------------------------------------------------------
def _sanitize_for_filename(s: str) -> str:
    s = s.strip().replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


def _default_output_csv(args, compile_flag: bool) -> str:
    model_tag = _sanitize_for_filename(args.model_name)
    qtag = "q" if args.qmodel_ckpt else "fp"
    kernel_tag = f"kernel-{args.quant_kernel_type}" if args.use_quant_kernels else "kernel-none"
    compile_tag = "compile" if compile_flag else "eager"
    zeus_tag = "zeus" if args.profile_energy else "nozeus"
    mem_tag = "mem" if args.profile_gpu_memory else "nomem"

    prompt_tag = _tag_from_list(args.prompt_tokens_list, "prompttok")
    new_tag = _tag_from_list(args.max_new_tokens_list, "new")
    bs_tag = _tag_from_list(args.batch_sizes_list, "bs")

    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = (f"decodebench_{model_tag}_{qtag}_{kernel_tag}_{compile_tag}"
             f"_dtype-{args.dtype}_seqlen-{args.seqlen}"
             f"{prompt_tag}{new_tag}{bs_tag}"
             f"_topk-{args.top_k}_temp-{args.temperature}"
             f"_{zeus_tag}_{mem_tag}_{ts}.csv")
    return os.path.join(".", fname)


def _get_pad_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def _get_fill_token_id(tokenizer) -> int:
    """A non-special token id used to extend prompts to a target token length."""
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    bad = {x for x in (eos, pad) if x is not None}

    for s in [" hello", " the", " a", ".", "0", "1", "and", "I", "to"]:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            continue
        for tid in reversed(ids):
            if tid not in bad:
                return int(tid)

    if eos is not None:
        return int(eos)
    if pad is not None:
        return int(pad)
    return 0


def _force_prompt_tokens(inputs, tokenizer, target_tokens: int):
    """
    Force batch to have exactly target_tokens tokens (truncate or extend).
    Expects a dict with input_ids (+ optional attention_mask/token_type_ids/position_ids).
    """
    if target_tokens <= 0:
        return inputs

    if "input_ids" not in inputs:
        raise KeyError("tokenizer output missing input_ids")

    input_ids = inputs["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError(f"Unexpected input_ids shape: {getattr(input_ids, 'shape', None)}")

    bsz, seqlen = input_ids.shape
    attn = inputs.get("attention_mask", None)
    if attn is None:
        attn = torch.ones((bsz, seqlen), dtype=torch.long, device=input_ids.device)
    else:
        attn = attn.to(device=input_ids.device)

    ttype = inputs.get("token_type_ids", None)
    pos = inputs.get("position_ids", None)

    if seqlen == target_tokens:
        inputs["input_ids"] = input_ids.to(dtype=torch.long)
        inputs["attention_mask"] = attn.to(dtype=torch.long)
        if ttype is not None and torch.is_tensor(ttype) and ttype.ndim == 2:
            inputs["token_type_ids"] = ttype.to(device=input_ids.device)
        if pos is not None and torch.is_tensor(pos) and pos.ndim == 2:
            inputs["position_ids"] = pos.to(device=input_ids.device)
        return inputs

    if seqlen > target_tokens:
        inputs["input_ids"] = input_ids[:, :target_tokens].to(dtype=torch.long)
        inputs["attention_mask"] = attn[:, :target_tokens].to(dtype=torch.long)
        if ttype is not None and torch.is_tensor(ttype) and ttype.ndim == 2:
            inputs["token_type_ids"] = ttype[:, :target_tokens]
        if pos is not None and torch.is_tensor(pos) and pos.ndim == 2:
            inputs["position_ids"] = torch.arange(target_tokens, device=input_ids.device,
                                                  dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        return inputs

    fill_id = _get_fill_token_id(tokenizer)
    need = target_tokens - seqlen
    filler = torch.full((bsz, need), fill_id, dtype=torch.long, device=input_ids.device)

    inputs["input_ids"] = torch.cat([input_ids.to(dtype=torch.long), filler], dim=1)
    inputs["attention_mask"] = torch.cat(
        [attn.to(dtype=torch.long),
         torch.ones((bsz, need), dtype=torch.long, device=input_ids.device)], dim=1)

    if ttype is not None and torch.is_tensor(ttype) and ttype.ndim == 2:
        inputs["token_type_ids"] = torch.cat(
            [ttype.to(device=input_ids.device),
             torch.zeros((bsz, need), dtype=ttype.dtype, device=input_ids.device)],
            dim=1,
        )

    if pos is not None and torch.is_tensor(pos) and pos.ndim == 2:
        inputs["position_ids"] = torch.arange(target_tokens, device=input_ids.device,
                                              dtype=torch.long).unsqueeze(0).expand(bsz, -1)

    return inputs


def _tokenize_bench_batch(tokenizer, texts: List[str], device: torch.device, prompt_tokens: int = 0):
    """
    Tokenize for benchmarking with guaranteed fixed-length, non-padded, rectangular batch.
    - If prompts differ in token length, raises (benchmarking wants fixed prefill length).
    - If prompt_tokens > 0, truncates/extends to exactly that many tokens.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    if "input_ids" not in inputs:
        raise RuntimeError("Tokenizer did not return input_ids.")

    input_ids = inputs["input_ids"]
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

    attn = inputs["attention_mask"].to(dtype=torch.long)
    lengths = attn.sum(-1).to(dtype=torch.long)

    if lengths.numel() == 0:
        raise RuntimeError("Empty tokenization result.")
    if not torch.all(lengths == lengths[0]):
        raise ValueError(f"Mixed prompt lengths in batch (lengths={lengths.tolist()}). "
                         "For benchmarking, use identical prompts (or a single --bench_prompt replicated).")

    true_len = int(lengths[0].item())
    if true_len <= 0:
        raise RuntimeError("Tokenized prompt length is 0.")

    # Strip padding: keep only real tokens
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and v.ndim == 2:
            inputs[k] = v[:, :true_len]

    # Now no padding remains: attention_mask all-ones
    inputs["attention_mask"] = torch.ones((len(texts), true_len), dtype=torch.long)

    if int(prompt_tokens) > 0:
        inputs = _force_prompt_tokens(inputs, tokenizer, int(prompt_tokens))

        # Ensure no pad tokens were introduced
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            ids = inputs["input_ids"]
            mask_pad = ids.eq(int(pad_id))
            if mask_pad.any():
                fill_id = _get_fill_token_id(tokenizer)
                ids = ids.clone()
                ids[mask_pad] = int(fill_id)
                inputs["input_ids"] = ids

        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    # Move to device + normalize dtypes
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
    inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
    return inputs


def _infer_model_max_positions(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    looks = ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]
    if cfg is not None:
        for key in looks:
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return int(v)
    v = getattr(model, "seqlen", None)
    if isinstance(v, int) and v > 0:
        return int(v)
    return None


# ------------------------------------------------------------
# Zeus helpers (robust across versions)
# ------------------------------------------------------------
def _make_zeus_monitor(args, output_csv: str):
    if not args.profile_energy:
        return None
    if ZeusMonitor is None:
        raise ImportError("Zeus not found. Install with: pip install zeus-ml")
    if not torch.cuda.is_available():
        raise RuntimeError("--profile_energy requires CUDA for this script.")

    if args.zeus_log_file is None or args.zeus_log_file.strip() == "":
        base = os.path.splitext(output_csv)[0]
        args.zeus_log_file = base + ".zeus_windows.csv"

    gpu_idx = torch.cuda.current_device()

    kwargs = {
        "gpu_indices": [gpu_idx],
        "cpu_indices": [],
        "log_file": args.zeus_log_file,
        "approx_instant_energy": bool(args.zeus_approx_instant_energy),
        "sync_execution_with": "torch",
    }
    try:
        sig = inspect.signature(ZeusMonitor)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        kwargs = {"gpu_indices": [gpu_idx], "cpu_indices": [], "log_file": args.zeus_log_file}

    return ZeusMonitor(**kwargs)


def _zeus_begin(monitor, key: str):
    if monitor is None:
        return
    try:
        monitor.begin_window(key, sync_execution=True)
    except TypeError:
        try:
            monitor.begin_window(key, sync_cuda=True)
        except TypeError:
            monitor.begin_window(key)


def _zeus_end(monitor, key: str, cancel: bool = False):
    if monitor is None:
        return None
    try:
        return monitor.end_window(key, sync_execution=True, cancel=cancel)
    except TypeError:
        pass
    try:
        return monitor.end_window(key, sync_cuda=True)
    except TypeError:
        pass
    try:
        return monitor.end_window(key)
    except Exception:
        return None


# ------------------------------------------------------------
# Sampling helpers (RNG outside the graph)
# ------------------------------------------------------------
def multinomial_sample_one_no_sync(probs_last: torch.Tensor) -> torch.Tensor:
    """Gumbel-max trick (no explicit CUDA sync). probs_last: [B, V] -> returns [B, 1] long."""
    q = torch.empty_like(probs_last).exponential_(1)
    return torch.argmax(probs_last / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs_last(
    logits_last: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Apply temperature+top_k then softmax. logits_last: [B, V]."""
    logits_last = logits_last / max(float(temperature), 1e-5)
    if top_k is not None:
        k = min(int(top_k), logits_last.size(-1))
        v, _ = torch.topk(logits_last, k)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits_last = torch.where(logits_last < pivot, -float("inf"), logits_last)
    return torch.nn.functional.softmax(logits_last, dim=-1)


def sample_from_last(logits_last: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs_last(logits_last, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_full(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None):
    """Preserve existing interface: sample from [B, T, V] → [:,-1]."""
    return sample_from_last(logits[:, -1], temperature, top_k)


# ------------------------------------------------------------
# (Optional) Compileable decode step: Capture only model forward (RNG excluded)
# ------------------------------------------------------------
def build_compiled_decode_step(model):
    @torch.no_grad()
    def decode_logits_step(cur_token: torch.Tensor, past_kv, cache_position: torch.Tensor):
        logits = model(
            cur_token.to(dtype=torch.long),
            past_key_values=past_kv,
            cache_position=cache_position.to(dtype=torch.long),
            use_cache=True,
        )[0]
        return logits[:, -1]  # [B, V]

    return torch.compile(
        decode_logits_step,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )


# ------------------------------------------------------------
# 1-token decode (eager)
# ------------------------------------------------------------
@torch.no_grad()
def decode_one_tokens(model, cur_token, past_kv, cache_position, top_k: int = 5, temperature: float = 0.6):
    logits = model(
        cur_token.to(dtype=torch.long),
        past_key_values=past_kv,
        cache_position=cache_position.to(dtype=torch.long),
        use_cache=True,
    )[0]
    next_token, _ = sample_full(logits, temperature=temperature, top_k=top_k)
    return next_token, logits


def make_cache(model, batch_size: int, max_cache_len: int, device, dtype):
    from transformers import StaticCache
    return StaticCache(model.config, batch_size, max_cache_len, layer_device_map=None, device=device, dtype=dtype)


def _sdpa_ctx(device):
    if device.type != "cuda":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except Exception:
        return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=True)


def _cleanup_cuda(device: torch.device):
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


# ------------------------------------------------------------
# Small format helpers (for printing summaries)
# ------------------------------------------------------------
def _to_float_or_none(x) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _fmt_gb(x) -> str:
    v = _to_float_or_none(x)
    if v is None:
        return "NA"
    return f"{v / 1e9:.3f} GB"


def _fmt_float(x, nd: int = 6) -> str:
    v = _to_float_or_none(x)
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


# ------------------------------------------------------------
# Generate (Correct per-iter peak mem + energy/token)
# ------------------------------------------------------------
@torch.no_grad()
def generate(
    model,
    tokenizer,
    text: Union[str, List[str]],
    max_new_tokens: int,
    top_k: int,
    callback,
    past_kv,
    temperature: float = 0.6,
    device=None,
    dtype=None,
    batch_size: int = 1,
    # profiling
    zeus_monitor=None,
    profile_gpu_memory: bool = False,
    # fixed prompt length
    prompt_tokens: int = 0,
    # compiled decode step (optional)
    compiled_decode_step=None,
):
    prof: Dict[str, Union[int, float, str]] = {}

    if device is None:
        device = next(model.parameters()).device

    # Build a batch:
    if isinstance(text, str):
        texts = [text] * int(batch_size)
    else:
        texts = list(text)
        if batch_size is not None and int(batch_size) != len(texts):
            raise ValueError(f"batch_size={batch_size} but got {len(texts)} prompts.")
        batch_size = len(texts)

    # FIX: reset peak stats before ANY per-iteration allocations on CUDA
    if profile_gpu_memory and device.type == "cuda":
        torch.cuda.synchronize(device)
        free_b, total_b = torch.cuda.mem_get_info(device)
        prof.update({
            "gpu_mem_free_before_bytes": int(free_b),
            "gpu_mem_total_bytes": int(total_b),
            "gpu_mem_alloc_before_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_mem_reserved_before_bytes": int(torch.cuda.memory_reserved(device)),
        })
        torch.cuda.reset_peak_memory_stats(device)

    inputs = _tokenize_bench_batch(tokenizer, texts, device=device, prompt_tokens=int(prompt_tokens))
    batch_size_inp, seq_length = inputs["input_ids"].shape
    if batch_size_inp != batch_size:
        raise RuntimeError(f"Tokenizer returned batch={batch_size_inp}, expected batch_size={batch_size}")

    pad_id = _get_pad_id(tokenizer)
    generated_ids = torch.full(
        (batch_size, seq_length + max_new_tokens),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    generated_ids[:, :seq_length] = inputs["input_ids"].to(dtype=torch.long)

    # Ensure cache is big enough AND batch-size compatible
    needed = int(seq_length + max_new_tokens)
    cache_bs = getattr(past_kv, "batch_size", batch_size) if past_kv is not None else batch_size
    cache_len = getattr(past_kv, "max_cache_len", 0) if past_kv is not None else 0

    if (past_kv is None) or (cache_len < needed) or (cache_bs != batch_size):
        past_kv = make_cache(model, batch_size, needed, device, dtype)
    else:
        if hasattr(past_kv, "reset"):
            past_kv.reset()

    cache_position_prefill = torch.arange(seq_length, device=device, dtype=torch.long)

    # --- Prefill ---
    _zeus_begin(zeus_monitor, "prefill")
    prefill_start = time.time()
    try:
        logits = model(
            **inputs,
            past_key_values=past_kv,
            cache_position=cache_position_prefill,
            use_cache=True,
        )[0]
    except Exception:
        _zeus_end(zeus_monitor, "prefill", cancel=True)
        raise
    prefill_wall = time.time() - prefill_start
    prefill_mes = _zeus_end(zeus_monitor, "prefill", cancel=False)

    prof["prefill_time_s_wall"] = float(prefill_wall)

    if prefill_mes is not None:
        prompt_tokens_total = int(batch_size * seq_length)
        prefill_energy = float(getattr(prefill_mes, "total_energy", 0.0))
        prefill_time = float(getattr(prefill_mes, "time", 0.0))
        prof.update({
            "prefill_time_s_zeus": prefill_time,
            "prefill_energy_j": prefill_energy,
            "prefill_avg_power_w": prefill_energy / max(prefill_time, 1e-12),
            "prefill_prompt_tokens_total": int(prompt_tokens_total),
            "prefill_energy_per_prompt_token_j": prefill_energy / max(prompt_tokens_total, 1),
        })

    # Token #1 (sample from prefill logits)
    next_token, _ = sample_full(logits, temperature=temperature, top_k=top_k)  # [B,1]
    generated_ids[:, seq_length] = next_token.squeeze(-1)
    callback(next_token)

    if max_new_tokens <= 1:
        to_decode = generated_ids[:, :seq_length + 1].to("cpu")
        texts_out = tokenizer.batch_decode(to_decode, skip_special_tokens=True)

        prof["decode_time_s_wall"] = 0.0
        prof["decode_tokens_total"] = 0
        prof["compiled_decode_failed"] = 0
        prof["compiled_decode_used"] = 0

        if profile_gpu_memory and device.type == "cuda":
            torch.cuda.synchronize(device)
            free_a, _ = torch.cuda.mem_get_info(device)
            peak_alloc = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            alloc_b = int(prof.get("gpu_mem_alloc_before_bytes", 0))
            res_b = int(prof.get("gpu_mem_reserved_before_bytes", 0))
            prof.update({
                "gpu_mem_free_after_bytes": int(free_a),
                "gpu_mem_alloc_after_bytes": int(torch.cuda.memory_allocated(device)),
                "gpu_mem_reserved_after_bytes": int(torch.cuda.memory_reserved(device)),
                "gpu_mem_peak_alloc_bytes": peak_alloc,
                "gpu_mem_peak_reserved_bytes": peak_reserved,
                "gpu_mem_peak_alloc_increase_bytes": int(max(0, peak_alloc - alloc_b)),
                "gpu_mem_peak_reserved_increase_bytes": int(max(0, peak_reserved - res_b)),
            })

        return generated_ids[:, :seq_length + 1], texts_out, 0.0, prof, past_kv

    # --- Decode loop ---
    cur_token = next_token
    cur_pos = int(seq_length)
    cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)

    _zeus_begin(zeus_monitor, "decode")
    decode_start = time.time()
    tokens_in_loop = max_new_tokens - 1  # token2..tokenN

    compiled_started = int(compiled_decode_step is not None)
    compiled_failed = 0

    with _sdpa_ctx(device):
        for _ in range(1, max_new_tokens):
            if compiled_decode_step is not None:
                try:
                    logits_last = compiled_decode_step(cur_token, past_kv, cache_position)  # [B,V]
                    next_token, _ = sample_from_last(logits_last, temperature=temperature, top_k=top_k)  # [B,1]
                except Exception as e:
                    print(f"[WARN] compiled decode failed; falling back to eager. {type(e).__name__}: {e}")
                    compiled_failed = 1
                    compiled_decode_step = None
                    next_token, _ = decode_one_tokens(model, cur_token, past_kv, cache_position, top_k=top_k,
                                                      temperature=temperature)
            else:
                next_token, _ = decode_one_tokens(model, cur_token, past_kv, cache_position, top_k=top_k,
                                                  temperature=temperature)

            cur_pos += 1
            generated_ids[:, cur_pos] = next_token.squeeze(-1)
            callback(next_token)

            cur_token = next_token
            cache_position += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_wall = time.time() - decode_start

    prof["decode_time_s_wall"] = float(decode_wall)

    decode_mes = _zeus_end(zeus_monitor, "decode", cancel=False)
    if decode_mes is not None:
        total_decode_tokens = int(batch_size * tokens_in_loop)
        decode_energy = float(getattr(decode_mes, "total_energy", 0.0))
        decode_time = float(getattr(decode_mes, "time", 0.0))
        prof.update({
            "decode_time_s_zeus": decode_time,
            "decode_energy_j": decode_energy,
            "decode_avg_power_w": decode_energy / max(decode_time, 1e-12),
            "decode_tokens_total": int(total_decode_tokens),
            "decode_energy_per_token_j": decode_energy / max(total_decode_tokens, 1),
        })
    else:
        prof["decode_tokens_total"] = int(batch_size * tokens_in_loop)

    # Peak GPU memory
    if profile_gpu_memory and device.type == "cuda":
        torch.cuda.synchronize(device)
        free_a, _ = torch.cuda.mem_get_info(device)
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        alloc_b = int(prof.get("gpu_mem_alloc_before_bytes", 0))
        res_b = int(prof.get("gpu_mem_reserved_before_bytes", 0))
        prof.update({
            "gpu_mem_free_after_bytes": int(free_a),
            "gpu_mem_alloc_after_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_mem_reserved_after_bytes": int(torch.cuda.memory_reserved(device)),
            "gpu_mem_peak_alloc_bytes": peak_alloc,
            "gpu_mem_peak_reserved_bytes": peak_reserved,
            "gpu_mem_peak_alloc_increase_bytes": int(max(0, peak_alloc - alloc_b)),
            "gpu_mem_peak_reserved_increase_bytes": int(max(0, peak_reserved - res_b)),
        })

    total_tokens = int(batch_size * tokens_in_loop)
    tps_total = float(total_tokens) / max(float(decode_wall), 1e-9)

    to_decode = generated_ids[:, :cur_pos + 1].to("cpu")
    texts_out = tokenizer.batch_decode(to_decode, skip_special_tokens=True)

    prof["compiled_decode_failed"] = int(compiled_failed)
    prof["compiled_decode_used"] = int(compiled_started and not compiled_failed)

    return generated_ids[:, :cur_pos + 1], texts_out, tps_total, prof, past_kv


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def _get_actual_prompt_lens(tokenizer, bench_prompt: str, prompt_tokens_list: List[int], device) -> Dict[int, int]:
    """Computes the actual sequence length for each prompt_tokens setting."""
    seq_len_map = {}
    for pt in sorted(set(int(x) for x in prompt_tokens_list)):
        warm_inputs = _tokenize_bench_batch(tokenizer, [bench_prompt], device=device, prompt_tokens=int(pt))
        seq_len_map[int(pt)] = int(warm_inputs["input_ids"].shape[1])
    return seq_len_map


def _validate_max_pos(model, seq_len_map: Dict[int, int], max_new_tokens_list: List[int]):
    """Validates if total sequence length exceeds model's maximum supported positions."""
    max_pos = _infer_model_max_positions(model)
    if max_pos is not None:
        for pt, seqlen_prompt in seq_len_map.items():
            for new in max_new_tokens_list:
                total_len = int(seqlen_prompt) + int(new)
                if total_len > int(max_pos):
                    raise ValueError(f"Invalid config: prompt_tokens={pt} + max_new_tokens={new} "
                                     f"exceeds model max positions {max_pos}.")


def _aggregate_config_rows(config_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a config:
      - Peak memory fields => MAX (not mean)
      - Throughput => total_tokens / total_time
      - Energy/token => total_energy / total_tokens (when available)
      - Other scalar metrics => mean if numeric
    """
    if not config_rows:
        return {}

    keys = list(config_rows[0].keys())
    agg = {k: "" for k in keys}
    agg["iter"] = "AGGREGATE"

    def _to_float(x) -> Optional[float]:
        if x in (None, ""):
            return None
        try:
            return float(str(x).strip())
        except Exception:
            return None

    # Totals for rate-style aggregation
    sum_tokens = 0.0
    sum_time = 0.0
    sum_energy = 0.0
    have_energy = False

    # Track max for peaks
    peak_fields = [
        "gpu_mem_peak_alloc_bytes",
        "gpu_mem_peak_reserved_bytes",
        "gpu_mem_peak_alloc_increase_bytes",
        "gpu_mem_peak_reserved_increase_bytes",
    ]
    for pf in peak_fields:
        mx = None
        for r in config_rows:
            v = _to_float(r.get(pf, ""))
            if v is None:
                continue
            mx = v if mx is None else max(mx, v)
        if mx is not None:
            agg[pf] = f"{mx:.0f}"

    # Sum tokens/time/energy for decode-level rates
    for r in config_rows:
        tok = _to_float(r.get("decode_tokens_total", ""))
        t = _to_float(r.get("decode_time_s", ""))
        e = _to_float(r.get("decode_energy_j", ""))
        if tok is not None and t is not None:
            sum_tokens += tok
            sum_time += t
        if e is not None and tok is not None and tok > 0:
            sum_energy += e
            have_energy = True

    if sum_time > 0 and sum_tokens > 0:
        agg["decode_tokens_total"] = f"{sum_tokens:.0f}"
        agg["decode_time_s"] = f"{sum_time:.6f}"
        agg["decode_tps_total"] = f"{(sum_tokens / sum_time):.6f}"
        bs0 = _to_float(config_rows[0].get("batch_size", ""))
        if bs0 is not None and bs0 > 0:
            agg["decode_tps_per_seq"] = f"{((sum_tokens / sum_time) / bs0):.6f}"

    if have_energy and sum_tokens > 0:
        agg["decode_energy_j"] = f"{sum_energy:.6f}"
        agg["decode_energy_per_token_j"] = f"{(sum_energy / sum_tokens):.9f}"

    # For other numeric fields: mean
    for k in keys:
        if k in ("iter", ) or k in peak_fields:
            continue
        if k in ("decode_tokens_total", "decode_time_s", "decode_tps_total", "decode_tps_per_seq", "decode_energy_j",
                 "decode_energy_per_token_j"):
            continue

        vals = []
        for r in config_rows:
            v = _to_float(r.get(k, ""))
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        agg[k] = f"{(sum(vals) / len(vals)):.6f}"

    return agg


def _print_config_summary(bs: int, pt: int, new: int, seqlen_prompt: int, agg_row: Dict[str, Any]):
    tps = agg_row.get("decode_tps_total", "")
    ept = agg_row.get("decode_energy_per_token_j", "")
    peak_alloc = agg_row.get("gpu_mem_peak_alloc_bytes", "")
    peak_reserved = agg_row.get("gpu_mem_peak_reserved_bytes", "")

    print(f"\n=== CONFIG SUMMARY: bs={bs}, pt={pt}, new={new} (seqlen_prompt={seqlen_prompt}) ===")
    print(f"AVG decode tokens/s (total): {_fmt_float(tps, nd=6)}")
    print(f"AVG decode energy/token (J): {_fmt_float(ept, nd=9)}")
    print(f"MAX peak GPU mem: alloc {_fmt_gb(peak_alloc)} | reserved {_fmt_gb(peak_reserved)}")
    print("============================================================\n")


def _print_summary_table(config_summaries: List[Dict[str, Any]]):
    if not config_summaries:
        return

    # sort for readability
    config_summaries = sorted(
        config_summaries, key=lambda r:
        (int(r["batch_size"]), int(r["prompt_tokens_target"]), int(r["max_new_tokens"])))

    headers = [
        "batch_size",
        "prompt_tokens_target",
        "seqlen_prompt",
        "max_new_tokens",
        "decode_tps_total",
        "decode_energy_per_token_j",
        "gpu_mem_peak_alloc_gb",
        "gpu_mem_peak_reserved_gb",
    ]

    # compute widths
    rows = []
    for r in config_summaries:
        rows.append([
            str(r.get("batch_size", "")),
            str(r.get("prompt_tokens_target", "")),
            str(r.get("seqlen_prompt", "")),
            str(r.get("max_new_tokens", "")),
            str(r.get("decode_tps_total", "NA")),
            str(r.get("decode_energy_per_token_j", "NA")),
            str(r.get("gpu_mem_peak_alloc_gb", "NA")),
            str(r.get("gpu_mem_peak_reserved_gb", "NA")),
        ])

    widths = []
    for j, h in enumerate(headers):
        mx = len(h)
        for row in rows:
            mx = max(mx, len(row[j]))
        widths.append(mx)

    def fmt_row(items: List[str]) -> str:
        return " | ".join(items[j].ljust(widths[j]) for j in range(len(items)))

    sep = "-+-".join("-" * w for w in widths)

    print("\n\n====================== CONFIG SUMMARY TABLE ======================")
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("==================================================================\n")


# ------------------------------------------------------------
# Benchmark driver (supports sweeps)
# ------------------------------------------------------------
def run_bench_sweep(model, tokenizer, compile_flag: bool, args):
    device = next(model.parameters()).device

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.iters <= 0:
        raise ValueError("Sweep benchmarking requires --iters > 0.")
    if args.streaming:
        raise ValueError("--streaming does not support sweeps.")

    if args.output_csv is None or args.output_csv.strip() == "":
        args.output_csv = _default_output_csv(args, compile_flag=compile_flag)

    zeus_monitor = _make_zeus_monitor(args, args.output_csv) if args.profile_energy else None

    # Pre-calculate prompt lengths and validate limits
    seq_len_by_pt = _get_actual_prompt_lens(tokenizer, args.bench_prompt, args.prompt_tokens_list, device)
    _validate_max_pos(model, seq_len_by_pt, args.max_new_tokens_list)

    # Compiled step memoization (OK); DO NOT memoize KV caches (huge; breaks mem stats across configs).
    compiled_step_by_key = {}
    compiled_enabled_by_key = {}

    def get_comp_step(bs: int, cache_len: int):
        if not compile_flag:
            return None
        k = (int(bs), int(cache_len))
        if k not in compiled_enabled_by_key:
            compiled_enabled_by_key[k] = True
            try:
                compiled_step_by_key[k] = build_compiled_decode_step(model)
            except Exception as e:
                compiled_enabled_by_key[k] = False
                print(f"[WARN] torch.compile failed for bs={bs}, cache_len={cache_len}: {e}")
        return compiled_step_by_key.get(k) if compiled_enabled_by_key.get(k) else None

    all_rows: List[Dict[str, Any]] = []
    config_summaries: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    for bs in map(int, args.batch_sizes_list):
        for pt, seqlen_prompt in seq_len_by_pt.items():
            for new in map(int, args.max_new_tokens_list):
                cache_len = int(seqlen_prompt + new)
                config_rows: List[Dict[str, Any]] = []

                if device.type == "cuda":
                    _cleanup_cuda(device)

                # One cache per config; reuse for warmup+iters; free after config.
                past_kv = make_cache(model, bs, cache_len, device=device, dtype=model.dtype)

                # Warmup + measurement runs
                total_runs = int(args.warmup) + int(args.iters)
                for run_idx in range(1, total_runs + 1):
                    is_measurement = (run_idx > int(args.warmup))
                    iter_num = run_idx - int(args.warmup)

                    step = get_comp_step(bs, cache_len)

                    # During warmup: DO NOT profile energy/mem (keeps logs clean + avoids overhead).
                    run_zeus = zeus_monitor if is_measurement else None
                    run_mem = bool(args.profile_gpu_memory) and is_measurement

                    _, _, tps_total, prof, past_kv = generate(
                        model=model,
                        tokenizer=tokenizer,
                        text=args.bench_prompt,
                        max_new_tokens=new,
                        top_k=int(args.top_k),
                        callback=lambda x: x,
                        past_kv=past_kv,
                        temperature=float(args.temperature),
                        device=device,
                        dtype=model.dtype,
                        batch_size=bs,
                        zeus_monitor=run_zeus,
                        profile_gpu_memory=run_mem,
                        prompt_tokens=pt,
                        compiled_decode_step=step,
                    )

                    if int(prof.get("compiled_decode_failed", 0)) == 1:
                        compiled_enabled_by_key[(bs, cache_len)] = False

                    if is_measurement:
                        # decode loop forwards only: token2..tokenN
                        tokens_in_loop = max(int(new) - 1, 0)
                        total_tokens = int(bs * tokens_in_loop)
                        decode_time_s = float(total_tokens) / max(float(tps_total), 1e-12) if total_tokens > 0 else 0.0

                        row = {
                            "iter": iter_num,
                            "model_name": args.model_name,
                            "qmodel_ckpt": args.qmodel_ckpt or "",
                            "batch_size": bs,
                            "prompt_tokens_target": pt,
                            "seqlen_prompt": seqlen_prompt,
                            "max_new_tokens": new,
                            "decode_tokens_per_seq": tokens_in_loop,
                            "decode_tokens_total": total_tokens,
                            "decode_time_s": f"{decode_time_s:.6f}",
                            "decode_tps_total": f"{float(tps_total):.6f}" if total_tokens > 0 else "",
                            "decode_tps_per_seq": f"{(float(tps_total) / bs):.6f}" if total_tokens > 0 else "",
                            "top_k": int(args.top_k),
                            "temperature": float(args.temperature),
                            "dtype": args.dtype,
                            "compiled": int(
                                bool(compile_flag and step is not None
                                     and int(prof.get("compiled_decode_used", 0)) == 1)),
                            "use_quant_kernels": int(bool(args.use_quant_kernels)),
                            "quant_kernel_type": args.quant_kernel_type if args.use_quant_kernels else "",
                            # Zeus (may be blank)
                            "prefill_time_s_zeus": f"{prof.get('prefill_time_s_zeus', '')}",
                            "prefill_energy_j": f"{prof.get('prefill_energy_j', '')}",
                            "prefill_energy_per_prompt_token_j": f"{prof.get('prefill_energy_per_prompt_token_j', '')}",
                            "decode_time_s_zeus": f"{prof.get('decode_time_s_zeus', '')}",
                            "decode_energy_j": f"{prof.get('decode_energy_j', '')}",
                            "decode_energy_per_token_j": f"{prof.get('decode_energy_per_token_j', '')}",
                            # Memory peaks (may be blank)
                            "gpu_mem_peak_alloc_bytes": f"{prof.get('gpu_mem_peak_alloc_bytes', '')}",
                            "gpu_mem_peak_reserved_bytes": f"{prof.get('gpu_mem_peak_reserved_bytes', '')}",
                            "gpu_mem_peak_alloc_increase_bytes": f"{prof.get('gpu_mem_peak_alloc_increase_bytes', '')}",
                            "gpu_mem_peak_reserved_increase_bytes": f"{prof.get('gpu_mem_peak_reserved_increase_bytes', '')}",
                        }
                        config_rows.append(row)

                        print(f"[iter {iter_num}/{args.iters}] bs={bs} pt={pt} new={new} tps={float(tps_total):.3f}")

                # Aggregate + REQUIRED: print per-config summary + store for end table
                if config_rows:
                    agg_row = _aggregate_config_rows(config_rows)

                    # Print after every config
                    _print_config_summary(bs=bs, pt=pt, new=new, seqlen_prompt=seqlen_prompt, agg_row=agg_row)

                    # Store summary row for final table (normalized formatting)
                    config_summaries.append({
                        "batch_size": bs,
                        "prompt_tokens_target": pt,
                        "seqlen_prompt": seqlen_prompt,
                        "max_new_tokens": new,
                        "decode_tps_total": _fmt_float(agg_row.get("decode_tps_total", ""), nd=6),
                        "decode_energy_per_token_j": _fmt_float(agg_row.get("decode_energy_per_token_j", ""), nd=9),
                        "gpu_mem_peak_alloc_gb": _fmt_gb(agg_row.get("gpu_mem_peak_alloc_bytes", "")),
                        "gpu_mem_peak_reserved_gb": _fmt_gb(agg_row.get("gpu_mem_peak_reserved_bytes", "")),
                    })

                    all_rows.extend(config_rows)
                    all_rows.append(agg_row)

                # Free cache for this config so it doesn't poison later configs
                del past_kv
                if device.type == "cuda":
                    _cleanup_cuda(device)

    # Save CSV
    if all_rows:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[OK] wrote CSV → {args.output_csv}")

    # REQUIRED: Final table at end of all configs
    _print_summary_table(config_summaries)


# ------------------------------------------------------------
# Interactive loop (kept single-config)
# ------------------------------------------------------------
def run_interactive(model, tokenizer, compile_flag: bool, args):
    device = next(model.parameters()).device
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.batch_sizes_list != [1]:
        raise ValueError("--streaming requires --batch_size=1")
    if args.prompt_tokens_list != [0] or args.max_new_tokens_list != [args.max_new_tokens_list[0]]:
        raise ValueError("--streaming does not support sweeps. Use --iters>0 and disable --streaming.")

    zeus_monitor = None
    if args.profile_energy:
        if args.output_csv is None or args.output_csv.strip() == "":
            args.output_csv = _default_output_csv(args, compile_flag=compile_flag)
        zeus_monitor = _make_zeus_monitor(args, args.output_csv)

    warm_inputs = _tokenize_bench_batch(tokenizer, [args.bench_prompt], device=device, prompt_tokens=0)
    warm_seq_len = int(warm_inputs["input_ids"].shape[1])
    past_kv = make_cache(model, 1, warm_seq_len + int(args.max_new_tokens_list[0]), device=device, dtype=model.dtype)

    compiled_step = None
    if compile_flag:
        try:
            compiled_step = build_compiled_decode_step(model)
        except Exception as e:
            compiled_step = None
            print(f"[WARN] torch.compile failed; continuing eager. {type(e).__name__}: {e}")

    while True:
        prompt = input("What is your prompt? ")
        if prompt.strip().lower() == "quit":
            break

        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        buffer = []
        period_id = tokenizer.encode(".")[-1]
        done_generating = False

        def stream_cb(x):
            nonlocal done_generating
            if done_generating:
                return
            buffer.append(tokenizer.decode([period_id] + x[0].tolist())[1:])
            if x[0].item() == tokenizer.eos_token_id:
                done_generating = True
            if len(buffer) == 4 or done_generating:
                print("".join(buffer), end="", flush=True)
                buffer.clear()

        ids, texts, decode_tps_total, prof, past_kv = generate(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_new_tokens=int(args.max_new_tokens_list[0]),
            top_k=int(args.top_k),
            callback=stream_cb,
            past_kv=past_kv,
            temperature=float(args.temperature),
            device=device,
            dtype=model.dtype,
            batch_size=1,
            zeus_monitor=zeus_monitor,
            profile_gpu_memory=bool(args.profile_gpu_memory),
            prompt_tokens=0,
            compiled_decode_step=compiled_step,
        )

        print(texts)
        print(f"\nDecoding throughput: {float(decode_tps_total):.02f} tokens/sec (total).\n")
        if args.profile_energy and prof.get("decode_energy_j") not in (None, ""):
            print(f"Decode energy: {float(prof['decode_energy_j']):.2f} J  "
                  f"(energy/token {float(prof.get('decode_energy_per_token_j', 0.0)):.6f} J)\n")
        if args.profile_gpu_memory and prof.get("gpu_mem_peak_alloc_bytes") not in (None, ""):
            print(f"Peak CUDA mem: alloc {int(prof['gpu_mem_peak_alloc_bytes'])/1e9:.3f} GB  "
                  f"reserved {int(prof['gpu_mem_peak_reserved_bytes'])/1e9:.3f} GB\n")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token-per-second benchmark sweep (torch.compile + Zeus)")

    # model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--qmodel_ckpt", type=str, default=None)

    # kernels
    parser.add_argument("--use_quant_kernels", type=str2bool, default=False)
    parser.add_argument("--quant_kernel_type", type=str, default="gemv", choices=["gemv", "gemm", "gemlite"])

    parser.add_argument("--bench_prompt", type=str, default="Follow the given instructions: ",
                        help="Fixed prompt for benchmarking (replicated for batch).")

    # Sweeps: accept single int or comma-separated list
    parser.add_argument(
        "--prompt_tokens", type=str, default="0",
        help="Input prompt tokens (prefill length). Single int or comma list. "
        "0 = natural token length of --bench_prompt.")
    parser.add_argument("--max_new_tokens", type=str, default="128",
                        help="Max new tokens to generate. Single int or comma list.")
    parser.add_argument("--batch_size", type=str, default="1",
                        help="Batch size. Single int or comma list (benchmark mode only).")

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="CSV path to store per-iter stats (auto-named if empty).")

    # utils
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # generation
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)

    # eval
    parser.add_argument("--ppl_task", type=str, default="c4,wikitext2")

    # compile / math
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (for debugging)")
    parser.add_argument("--disable_tf32", action="store_true")

    # profiling
    parser.add_argument("--profile_gpu_memory", action="store_true",
                        help="Record peak CUDA memory usage (alloc/reserved) per iteration.")
    parser.add_argument("--profile_energy", action="store_true",
                        help="Record energy/time via ZeusMonitor (requires: pip install zeus-ml).")
    parser.add_argument("--zeus_log_file", type=str, default=None,
                        help="Zeus window log CSV path (defaults to <output_csv>.zeus_windows.csv).")
    parser.add_argument("--zeus_approx_instant_energy", action="store_true",
                        help="Approx energy for very short windows where NVML energy counter reads 0J.")

    args = parser.parse_args()

    # Resolve sweeps
    args.prompt_tokens_list = _parse_int_list(args.prompt_tokens, "prompt_tokens")
    args.max_new_tokens_list = _parse_int_list(args.max_new_tokens, "max_new_tokens")
    args.batch_sizes_list = _parse_int_list(args.batch_size, "batch_size")

    # utils
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print(f"Using CUDA device: {device}")

    # load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    # load model
    if args.qmodel_ckpt:
        if "aqlm" in args.qmodel_ckpt.lower():
            model = AutoModelForCausalLM.from_pretrained(
                args.qmodel_ckpt,
                device_map=device,
                dtype="auto",
                low_cpu_mem_usage=True,
            )
            model.seqlen = args.seqlen
        else:
            model = load_compressed_model(
                model_name_or_path=args.model_name,
                checkpoint_path=args.qmodel_ckpt,
                seqlen=args.seqlen,
                device=str(device),
                dtype=dtype,
            )
            if args.use_quant_kernels:
                from rich.progress import track
                print(f"Using custom CUDA {dtype} {args.quant_kernel_type.upper()} kernels...")
                nano_modules = [module for module in model.modules() if isinstance(module, NanoQuantLinear)]
                for module in track(nano_modules, description="[cyan]Preparing kernels..."):
                    module.to(device)
                    module._prepare_kernel(kernel_type=args.quant_kernel_type, dtype=dtype)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    model.seqlen = args.seqlen
    model.to(device)
    model.eval()

    # inference
    if not args.disable_tf32:
        torch.set_float32_matmul_precision("high")

    # ppl evaluation (runs once)
    if len(args.ppl_task) > 0:
        from ..utils.data_utils import get_test_loaders
        from ..utils.eval_utils import evaluate_ppl

        datasets = [ds.strip() for ds in args.ppl_task.split(",") if ds.strip()]
        for dataset in datasets:
            try:
                _, testloader = get_test_loaders(dataset, model_name=model.config._name_or_path, seqlen=model.seqlen)
                _ = evaluate_ppl(model, testloader, device, dataset, args=None, verbose=True)
            except Exception as e:
                print(f"Failed to evaluate PPL on dataset {dataset}: {e}")
                continue

    compile_flag = not args.no_compile

    # Run
    if args.streaming and args.iters == 0:
        run_interactive(model, tokenizer, compile_flag=compile_flag, args=args)
    else:
        run_bench_sweep(model, tokenizer, compile_flag=compile_flag, args=args)
