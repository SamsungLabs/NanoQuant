# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for NanoQuant model compression and evaluation.

Usage:
    python -m nanoquant.main --model_id meta-llama/Llama-2-7b-hf --qmodel_path output.pt
    nanoquant --model_id meta-llama/Llama-2-7b-hf --qmodel_path output.pt
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

try:
    from loguru import logger
except ImportError:
    import logging as logger

import torch
from transformers import HfArgumentParser

from .modules.hub import NanoQuantConfigDataclass, NanoQuantModel
from .utils.eval_utils import evaluate_model
from .utils.load_utils import load_tokenizer
from .utils.utils import cleanup_memory


@dataclass
class ModelArguments:
    model_id: str = field(
        default="Qwen/Qwen3-4B-Base",
        metadata={"help": "Model identifier or local path"},
    )
    seqlen: int = field(default=2048, metadata={"help": "Sequence length"})
    qmodel_path: Optional[str] = field(default=None, metadata={"help": "Path to save/load quantized model checkpoint"})
    from_hub: bool = field(default=False, metadata={"help": "Load pre-quantized model from HuggingFace Hub"})
    hub_model_id: Optional[str] = field(default=None,
                                        metadata={"help": "HuggingFace Hub model ID (defaults to model_id)"})
    device_map: str = field(
        default="cpu",
        metadata={"help": "Device map for model loading ('cpu' or 'auto')"},
    )


@dataclass
class QuantArguments:
    bits: float = field(default=1.0, metadata={"help": "Target quantization bits"})
    seed: int = field(default=0, metadata={"help": "Random seed"})
    num_calib_samples: int = field(default=128, metadata={"help": "Number of calibration samples"})
    calib_dataset: str = field(default="wikitext2", metadata={"help": "Calibration dataset"})
    calib_shrinkage: float = field(default=0.4, metadata={"help": "Calibration shrinkage factor"})
    calib_strategy: str = field(
        default="online",
        metadata={
            "help": "Calibration strategy",
            "choices": ["online", "two_phase", "dbf", "none"],
        },
    )


@dataclass
class TuneArguments:
    tune_nonfact: bool = field(default=True, metadata={"help": "Tune non-factorized layers"})
    nonfact_lr: float = field(default=1e-4, metadata={"help": "LR for non-factorized binary parameters"})
    nonfact_batch_size: int = field(default=4, metadata={"help": "Batch size for non-factorized tuning"})
    nonfact_epochs: int = field(default=8, metadata={"help": "Epochs for non-factorized tuning"})
    admm_type: str = field(
        default="nanoquant",
        metadata={
            "help": "ADMM type",
            "choices": ["nanoquant", "dbf"]
        },
    )
    admm_outer_iters: int = field(default=400, metadata={"help": "ADMM outer iterations"})
    admm_inner_iters: int = field(default=5, metadata={"help": "ADMM inner iterations"})
    admm_reg: float = field(default=3e-2, metadata={"help": "ADMM regularization strength"})
    admm_penalty_scheduler: str = field(
        default="linear",
        metadata={
            "help": "ADMM penalty scheduler",
            "choices": ["linear", "cubic", "logistic", "exp_decay", "exp_growth"],
        },
    )
    admm_print_steps: bool = field(default=False, metadata={"help": "Print ADMM optimization steps"})
    tune_fact: bool = field(default=True, metadata={"help": "Tune factorized layers"})
    fact_binary_lr: float = field(default=1e-5, metadata={"help": "LR for factorized binary parameters"})
    fact_scale_lr: float = field(default=1e-5, metadata={"help": "LR for factorized scale parameters"})
    fact_bias_lr: float = field(default=1e-5, metadata={"help": "LR for factorized bias parameters"})
    fact_batch_size: int = field(default=1, metadata={"help": "Batch size for factorized tuning"})
    fact_epochs: int = field(default=8, metadata={"help": "Epochs for factorized tuning"})
    tune_model: bool = field(default=True, metadata={"help": "Perform model-level KD tuning"})
    model_kd_lr: float = field(default=1e-5, metadata={"help": "LR for model knowledge distillation"})
    model_kd_batch_size: int = field(default=1, metadata={"help": "Batch size for model KD"})


@dataclass
class EvalArguments:
    ppl_task: str = field(
        default="",
        metadata={"help": "Perplexity dataset(s), comma-separated. Leave empty to skip."},
    )
    zeroshot_task: str = field(
        default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge",
        metadata={"help": "Zero-shot tasks, comma-separated. Leave empty to skip."},
    )
    batch_size: str = field(default="auto", metadata={"help": "Batch size for zero-shot evaluation (auto = automatic)"})
    num_fewshot: int = field(default=0, metadata={"help": "Few-shot examples for zero-shot tasks"})
    limit: int = field(default=-1, metadata={"help": "Sample limit for zero-shot (-1 = all)"})


def init_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    if hasattr(logger, "remove"):
        try:
            logger.remove()
        except ValueError:
            pass
        logger.add(
            sys.stderr,
            level=log_level,
            format=
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(log_file, level="DEBUG", rotation="10 MB")
    else:
        logger.basicConfig(level=getattr(logger, log_level, logger.INFO))


def main():
    parser = HfArgumentParser((ModelArguments, QuantArguments, TuneArguments, EvalArguments))
    model_args, quant_args, tune_args, eval_args = parser.parse_args_into_dataclasses()

    init_logging()

    # Merge into NanoQuantConfigDataclass
    quant_config = NanoQuantConfigDataclass(
        model_id=model_args.model_id,
        bits=quant_args.bits,
        seed=quant_args.seed,
        num_calib_samples=quant_args.num_calib_samples,
        calib_dataset=quant_args.calib_dataset,
        calib_shrinkage=quant_args.calib_shrinkage,
        calib_strategy=quant_args.calib_strategy,
        seqlen=model_args.seqlen,
        device_map=model_args.device_map,
        tune_nonfact=tune_args.tune_nonfact,
        nonfact_lr=tune_args.nonfact_lr,
        nonfact_batch_size=tune_args.nonfact_batch_size,
        nonfact_epochs=tune_args.nonfact_epochs,
        admm_type=tune_args.admm_type,
        admm_outer_iters=tune_args.admm_outer_iters,
        admm_inner_iters=tune_args.admm_inner_iters,
        admm_reg=tune_args.admm_reg,
        admm_penalty_scheduler=tune_args.admm_penalty_scheduler,
        admm_print_steps=tune_args.admm_print_steps,
        tune_fact=tune_args.tune_fact,
        fact_binary_lr=tune_args.fact_binary_lr,
        fact_scale_lr=tune_args.fact_scale_lr,
        fact_bias_lr=tune_args.fact_bias_lr,
        fact_batch_size=tune_args.fact_batch_size,
        fact_epochs=tune_args.fact_epochs,
        tune_model=tune_args.tune_model,
        model_kd_lr=tune_args.model_kd_lr,
        model_kd_batch_size=tune_args.model_kd_batch_size,
    )

    if model_args.from_hub:
        hub_id = model_args.hub_model_id or model_args.model_id
        logger.info(f"Loading pre-quantized model from Hub: {hub_id}")
        nanoquant_model = NanoQuantModel.from_pretrained(hub_id, dtype=torch.bfloat16, device_map="cuda")
        loaded_from_hub = True
    else:
        logger.info(f"Quantizing model: {model_args.model_id}")
        nanoquant_model = NanoQuantModel.from_pretrained_quantize(
            model_id=model_args.model_id,
            qmodel_path=model_args.qmodel_path,
            quant_config=quant_config,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        loaded_from_hub = False

    model = nanoquant_model.model
    cleanup_memory()

    if model_args.qmodel_path and not os.path.exists(model_args.qmodel_path) and not loaded_from_hub:
        nanoquant_model._save_checkpoint(model, model_args.qmodel_path)
        logger.info(f"Saved quantized model to {model_args.qmodel_path}")

    model.eval()
    if not loaded_from_hub:
        model = model.cuda()

    tokenizer = load_tokenizer(model_args.model_id)

    try:
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks_str=eval_args.zeroshot_task,
            eval_ppl=eval_args.ppl_task,
            num_fewshot=eval_args.num_fewshot,
            limit=eval_args.limit,
            batch_size="auto" if eval_args.batch_size is None else eval_args.batch_size,
        )
        logger.info(f"Results:\n{json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise RuntimeError(f"Evaluation failed: {e}") from e


if __name__ == "__main__":
    main()
