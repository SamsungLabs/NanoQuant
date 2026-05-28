# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

def NanoQuantConfig(
    # model id
    model_id: str = "meta-llama/Llama-2-7b-hf",
    # quant precision
    bits: float = 1.0,
    # calib
    seed: int = 0,
    num_calib_samples: int = 128,
    calib_dataset: str = "wikitext2",
    calib_shrinkage: float = 0.4,
    calib_strategy: str = "online",
    seqlen: int = 2048,
    device_map: str = "cpu",
    # tune_nonfact
    tune_nonfact: bool = True,
    nonfact_lr: float = 1e-4,
    nonfact_batch_size: int = 4,
    nonfact_epochs: int = 8,
    # fact (admm)
    admm_type: str = "nanoquant",
    admm_outer_iters: int = 400,
    admm_inner_iters: int = 5,
    admm_reg: float = 3e-2,
    admm_penalty_scheduler: str = "linear",
    admm_print_steps: bool = False,
    # tune_fact
    tune_fact: bool = True,
    fact_binary_lr: float = 1e-5,
    fact_scale_lr: float = 1e-5,
    fact_bias_lr: float = 1e-5,
    fact_batch_size: int = 1,
    fact_epochs: int = 8,
    # tune_model
    tune_model: bool = True,
    model_kd_lr: float = 1e-5,
    model_kd_batch_size: int = 1,
    model_kd_epochs: int = 8,
) -> dict:
    return {
        # model id
        "model_id": model_id,
        # quant precision
        "bits": bits,
        # calibration
        "seed": seed,
        "num_calib_samples": num_calib_samples,
        "calib_dataset": calib_dataset,
        "calib_shrinkage": calib_shrinkage,
        "calib_strategy": calib_strategy,
        "seqlen": seqlen,
        "device_map": device_map,
        # tune_nonfact
        "tune_nonfact": tune_nonfact,
        "nonfact_lr": nonfact_lr,
        "nonfact_batch_size": nonfact_batch_size,
        "nonfact_epochs": nonfact_epochs,
        # fact (admm)
        "admm_type": admm_type,
        "admm_outer_iters": admm_outer_iters,
        "admm_inner_iters": admm_inner_iters,
        "admm_reg": admm_reg,
        "admm_penalty_scheduler": admm_penalty_scheduler,
        'admm_print_steps': admm_print_steps,
        # tune_fact
        "tune_fact": tune_fact,
        "fact_binary_lr": fact_binary_lr,
        "fact_scale_lr": fact_scale_lr,
        "fact_bias_lr": fact_bias_lr,
        "fact_batch_size": fact_batch_size,
        "fact_epochs": fact_epochs,
        # tune_model
        "tune_model": tune_model,
        "model_kd_lr": model_kd_lr,
        "model_kd_batch_size": model_kd_batch_size,
        "model_kd_epochs": model_kd_epochs,
    }
