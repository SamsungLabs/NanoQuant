# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig)

from ..kernel.utils import binary_unpacker
from ..utils.utils import cleanup_memory, get_decoder_layers


def load_model(model_id, seqlen=2048, device_map="cpu"):
    """
    Loads a pretrained model from the Hugging Face Hub and resizes positional embeddings if needed.
    
    For large models (>70B), use device_map="auto" with max_memory for GPU+CPU offloading.
    """
    def skip(*args, **kwargs):
        pass

    nn.init.kaiming_uniform_ = skip
    nn.init.uniform_ = skip
    nn.init.normal_ = skip

    # load model from huggingface
    print(f"Loading model '{model_id}'...")

    config = None
    if "mobilellm" in model_id.lower():  # have to adjust config for mobilellm, to enable lm_head
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.share_embedding = False

    # load model with device_map
    if device_map == "cpu":
        # Default: load to CPU for calibration and training
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                     trust_remote_code=True, device_map={'': 'cpu'})
    else:
        # For large models: load with specified device_map (e.g., "auto" for GPU+CPU offloading)
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                     trust_remote_code=True, device_map=device_map,
                                                     max_memory={0: '80GiB'})

    print(model)

    # disable kv cache
    model.config.use_cache = False

    # Set and potentially resize sequence length and positional embeddings
    original_seqlen = model.config.max_position_embeddings
    if not hasattr(model, "seqlen"):
        setattr(model, "seqlen", original_seqlen)

    if seqlen != -1 and seqlen > original_seqlen:
        print(f"Resizing model's position embeddings from {original_seqlen} to {seqlen}.")
        model.config.max_position_embeddings = seqlen
        model.seqlen = seqlen

        if model.config.model_type == "opt":
            offset = model.model.decoder.embed_positions.offset
            new_num_positions = seqlen + offset

            old_embed_positions = model.model.decoder.embed_positions
            old_num_positions, embedding_dim = old_embed_positions.weight.shape

            if new_num_positions > old_num_positions:
                new_embed_positions = nn.Embedding(new_num_positions, embedding_dim)

                init_std = getattr(model.config, 'init_std', 0.02)
                new_embed_positions.weight.data.normal_(mean=0.0, std=init_std)

                new_embed_positions.weight.data[:old_num_positions, :] = old_embed_positions.weight.data

                model.model.decoder.embed_positions = new_embed_positions
                print(f"Resized 'model.decoder.embed_positions' from {old_num_positions} to {new_num_positions}.")

        elif model.config.model_type in ["llama", "mistral", "mixtral"] or model.config.model_type.startswith("gemma"):
            print(
                f"Model type is {model.config.model_type} which uses RoPE. `max_position_embeddings` in config updated. No learned embedding resize needed."
            )

        else:
            print(
                f"Warning: Sequence length resizing for model type '{model.config.model_type}' is not explicitly handled. "
                "This may cause issues if the model uses learned positional embeddings.")

    elif seqlen != -1:
        model.config.max_position_embeddings = seqlen
        model.seqlen = seqlen

    return model


def load_tokenizer(model_name):
    """
    Returns the tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    gen_cfg = GenerationConfig.from_pretrained(model_name)

    def resolve_id(token_id):
        return token_id if isinstance(token_id, int) else token_id[0]

    tokenizer.bos_token_id = resolve_id(gen_cfg.bos_token_id)
    tokenizer.eos_token_id = resolve_id(gen_cfg.eos_token_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def cache_inputs_and_kwargs(model, dataloader, dev):
    """Captures and caches inputs for the first layer."""
    print("Caching initial inputs & kwargs using Catcher...")
    n_samples = len(dataloader)
    dtype = torch.bfloat16
    model_type = model.config.model_type
    layers = get_decoder_layers(model)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            cache['inputs'][cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['kwargs'] is None:
                # Cache all kwargs, including position embeddings for Gemma3
                cache['kwargs'] = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        cache['kwargs'][k] = v.cpu()
                    else:
                        cache['kwargs'][k] = v
            raise ValueError

        def __getattr__(self, name: str):
            """Forward attribute access to the wrapped module to support model-specific attributes like attention_type."""
            # Forward attribute access to the wrapped module
            if name != 'module':
                return getattr(self.module, name)
            # Default behavior for 'module' attribute
            return super().__getattr__(name)

    if model_type == "opt":
        model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions.to(dev)
    else:
        model.model.embed_tokens.to(dev)
    layers[0].to(dev)

    cache = {
        'inputs': torch.zeros((n_samples, model.seqlen, model.config.hidden_size), dtype=dtype),
        'kwargs': None,
        'i': 0
    }
    layers[0] = Catcher(layers[0])

    print("Capturing inputs...")
    for i in range(n_samples):
        try:
            model(dataloader[i].unsqueeze(0).to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0].cpu()

    for key in ["embed_tokens", "embed_positions", "norm", "rotary_emb", "rotary_emb_local"]:
        if hasattr(model.model, key) and getattr(model.model, key, None) is not None:
            getattr(model.model, key).cpu()

    cleanup_memory(verbose=False)
    kwargs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cache['kwargs'].items()}
    print("Initial input caching finished.")
    return cache['inputs'], kwargs


def get_embeddings(model: nn.Module) -> List[nn.Module]:
    """
    Helper to locate the embedding layers of the model.
    """
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            return [model.model.embed_tokens]
        if hasattr(model.model, 'decoder'):
            # OPT specific handling
            embeddings = []
            if hasattr(model.model.decoder, 'embed_tokens'):
                embeddings.append(model.model.decoder.embed_tokens)
            if hasattr(model.model.decoder, 'embed_positions'):
                embeddings.append(model.model.decoder.embed_positions)
            return embeddings
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return [model.transformer.wte]
    return []


class ParameterWrapper(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = nn.ParameterList(params)


def get_compressed_state_dict(model: nn.Module):
    """
    Generates a state_dict, forcing NanoQuantLinear modules to use their
    custom packing logic.
    """
    from ..modules.linear import NanoQuantLinear

    final_state_dict = OrderedDict()

    # 1. First add parameters from modules that are not custom layers
    for name, param in model.named_parameters():
        module_path = name.rsplit('.', 1)[0]
        try:
            module = model.get_submodule(module_path)
            if not isinstance(module, NanoQuantLinear):
                final_state_dict[name] = param.data
        except AttributeError:
            final_state_dict[name] = param.data

    # 2. Iterate through custom modules and add using custom state_dict
    for name, mod in model.named_modules():
        if isinstance(mod, NanoQuantLinear):
            module_state_dict = mod.state_dict(prefix=name + '.')
            final_state_dict.update(module_state_dict)

    return final_state_dict


def _load_and_process_state_dict(checkpoint_path: str, dtype: torch.dtype) -> Dict[str, Any]:
    t0 = time.time()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find model weights at {checkpoint_path}")

    # torch.load with best-effort fast/safe options (mmap/weights_only when available)
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "mmap" in sig.parameters:
        kwargs["mmap"] = True  # faster / lower peak RAM
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = True  # default True since 2.6 when pickle_module not passed

    try:
        sd = torch.load(checkpoint_path, **kwargs)
    except Exception:
        if kwargs.get("weights_only", False):
            kwargs["weights_only"] = False  # ONLY if you trust the checkpoint
            sd = torch.load(checkpoint_path, **kwargs)
        else:
            raise

    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict dict from {checkpoint_path}, got {type(sd)}")

    if not any(k.endswith("_packed") for k in sd):
        print("INFO: No packed weights found. Loading weights as is.")
        return sd

    print("INFO: Packed format detected. Unpacking weights...")

    packed = {k: v for k, v in sd.items() if k.endswith("_packed")}
    shapes = {k: v for k, v in sd.items() if k.endswith("_shape")}
    out = {k: v for k, v in sd.items() if not (k.endswith("_packed") or k.endswith("_shape"))}

    for pk, pv in packed.items():
        prefix, packed_name = pk.rsplit(".", 1)
        base = packed_name[:-7]  # strip "_packed"
        sk = f"{prefix}.{base}_shape"
        st = shapes.get(sk, None)
        if st is None:
            continue
        shape = tuple(int(x) for x in (st.tolist() if isinstance(st, torch.Tensor) else st))
        out[f"{prefix}.{base}"] = binary_unpacker(pv, shape).to(dtype)

    del packed, shapes
    cleanup_memory()

    print(f"INFO: Unpacking took {time.time() - t0:.2f}s.")
    return out


def load_compressed_model(model_name_or_path: str, checkpoint_path: str, seqlen: int, device: str, has_mid_scale=False,
                          dtype=torch.bfloat16):
    t0 = time.time()
    print(f"INFO: Loading model config from '{model_name_or_path}' and weights from '{checkpoint_path}'.")
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Build model with empty/meta init if possible (saves time/RAM)
    meta_init = False
    try:
        from accelerate import init_empty_weights  # type: ignore
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        meta_init = True
    except Exception:
        try:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
            meta_init = True
        except Exception:
            model = AutoModelForCausalLM.from_config(config)

    sd = _load_and_process_state_dict(checkpoint_path, dtype)

    def convert_layers(m: nn.Module):
        from ..modules.linear import NanoQuantLinear
        for name, module in m.named_modules():
            if type(module) is nn.Linear and "lm_head" not in name:
                base = f"{name}."
                if (base + "V") in sd or (base + "U") in sd:
                    module.__class__ = NanoQuantLinear
                    rank = sd[base + "V"].shape[0]
                    module.init_for_inference(rank=rank, has_scale_mid=has_mid_scale)

    print("INFO: Converting layers to compressed format...")
    convert_layers(model)

    print("INFO: Loading weights into the model...")
    if meta_init:
        try:
            model.load_state_dict(sd, strict=False, assign=True)
        except TypeError:
            # assign=True unsupported -> guaranteed fallback path
            model = AutoModelForCausalLM.from_config(config)
            convert_layers(model)
            model.load_state_dict(sd, strict=False)
            meta_init = False
    else:
        model.load_state_dict(sd, strict=False)

    # Handle tied / missing lm_head weights (common when output head tied to embeddings)
    if meta_init and hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    if meta_init and hasattr(model, "lm_head") and getattr(getattr(model.lm_head, "weight", None), "is_meta", False):
        # Materialize lm_head on CPU (to_empty if available; else allocate)
        if hasattr(model.lm_head, "to_empty") and callable(model.lm_head.to_empty):
            model.lm_head.to_empty(device="cpu")
        else:
            w = model.lm_head.weight
            model.lm_head.weight = nn.Parameter(torch.empty(w.shape, device="cpu", dtype=dtype), requires_grad=True)

        # Tie again; if tie_weights logic changes, force alias to embeddings
        try:
            model.tie_weights()
        except Exception:
            pass
        if getattr(model.lm_head.weight, "is_meta", False) and hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and not getattr(emb.weight, "is_meta", False):
                model.lm_head.weight = emb.weight

        # Bias (if present) must not remain meta
        if getattr(getattr(model.lm_head, "bias", None), "is_meta", False):
            model.lm_head.bias = nn.Parameter(torch.zeros(model.lm_head.weight.shape[0], device="cpu", dtype=dtype))

    if meta_init:
        leftover = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
        if leftover:
            raise RuntimeError("Model still has meta parameters after loading (missing weights). "
                               f"Example keys: {leftover[:10]}")

    del sd
    cleanup_memory()

    # print the size of parameters' GB in named_modules
    total_gb = 0
    for param in model.parameters():
        total_gb += param.nelement() * param.element_size() / (1024**3)
    print(f"Loaded compressed model size: {total_gb:.2f} GB")

    model.seqlen = seqlen if seqlen != -1 else config.max_position_embeddings
    model.config._attn_implementation = "sdpa"
    model.eval()
    print(f"model.seqlen={model.seqlen}")
    print(f"Compressed model successfully loaded to {device} in {time.time() - t0:.2f}s")
    return model
