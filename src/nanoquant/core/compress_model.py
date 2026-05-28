# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import random

import torch
import torch.nn.functional as F
from ..optimi import AdamW
from ..core.compress_block import (factorize_and_replace, tune_fact, tune_nonfact)
from ..modules.linear import NanoQuantLinear
from ..utils.eval_utils import evaluate_ppl_after_block
from ..utils.load_utils import cache_inputs_and_kwargs, load_tokenizer
from ..utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers, get_layers_to_factorize,
                           set_seed)
from tqdm import tqdm, trange


@torch.no_grad()
def compress_block_recon(model, fp_model, dataloader, quant_config):
    """
    Compresses a model using a functional, sequential tune-then-factorize approach.
    """
    # set seed
    set_seed(quant_config['seed'])
    # get device
    dev = "cuda"
    # adjust model configs
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = False
    # adjust fp model configs
    fp_model.gradient_checkpointing_disable()
    fp_model.eval()
    fp_model.config.use_cache = False
    # get relevant blocks/layers
    q_blocks = get_decoder_layers(model)
    fp_blocks = get_decoder_layers(fp_model)
    layers_to_factorize = get_layers_to_factorize(model.config.model_type)
    # get admm ranks
    admm_ranks = calculate_ranks(model, layers_to_factorize, quant_config)
    # get kwargs
    original_inputs, kwargs = cache_inputs_and_kwargs(fp_model, dataloader, dev)
    kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs['use_cache'] = False
    if 'past_key_value' in kwargs:
        kwargs['past_key_value'] = None
    # get inputs
    compressed_inputs = original_inputs.clone().detach().cpu()

    # block reconstruction loop
    for i in trange(len(q_blocks), desc="Compressing Layers"):
        cleanup_memory()
        # move qblock and fp_block to gpu
        q_block = q_blocks[i].to(dev)
        fp_block = fp_blocks[i].to(dev)
        # Calculate target outputs in batches to minimize CPU-GPU transfers
        with torch.no_grad():
            target_outputs = torch.zeros_like(original_inputs)
            for j in range(quant_config['num_calib_samples']):
                batch_input = original_inputs[j:j + 1].to(dev)
                batch_output = fp_block(batch_input, **kwargs)[0]
                target_outputs[j:j + 1] = batch_output.cpu().detach()
        # get qblock inputs
        tuning_inputs = compressed_inputs.clone().detach()
        # get all linear layers
        sublayers = find_layers(q_block)
        # get importance
        # Try to get importance from common layer names, fall back to uniform
        importance_layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
        if importance_layer is None:
            # Fallback to uniform importance if expected layer not found
            importance = torch.ones(model.config.hidden_size, device=dev)
        elif not hasattr(importance_layer, 'o_norm'):
            # Fallback if o_norm attribute missing
            importance = torch.ones(model.config.hidden_size, device=dev)
        else:
            importance = importance_layer.o_norm.to(dev)
        # move data to GPU
        tuning_inputs = tuning_inputs.to(dev)
        target_outputs = target_outputs.to(dev)
        # compress each linear layer
        for name in layers_to_factorize:
            if name not in sublayers: continue
            # 1/3) tune non-factorized, full-precision weights to absorb quant error
            if quant_config['tune_nonfact']:
                print(f"\t(1/3) Block {i+1}/{len(q_blocks)}, {name} | Tuning Non-Factorized Weights...")
                tune_nonfact(q_block, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                cleanup_memory()
            # 2/3) ADMM to factorize/initialize low-rank binary matrices and scales
            print(f"\t(2/3) Block {i+1}/{len(q_blocks)}, {name} | Initialization via ADMM...")
            curr_rank = admm_ranks.get(f"{i}.{name}")
            nano_linear, final_factor_results = factorize_and_replace(q_block, name, curr_rank, quant_config)
            del final_factor_results
            cleanup_memory()
            # 3/3) tune low-rank binary and scales
            if quant_config['tune_fact']:
                print(f"\t(3/3) Block {i+1}/{len(q_blocks)}, {name} | Tuning Factorized Weights...")
                tune_fact(q_block, nano_linear, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                cleanup_memory()
            cleanup_memory()

        # move fp_blocks[i] to cpu
        fp_blocks[i] = fp_block.cpu()
        # fp_blocks[i+1] input = fp_blocks[i] output
        original_inputs = target_outputs.clone().detach().cpu()

        # use qblock[i] outputs for qblocks[i+1] inputs
        with torch.no_grad():
            for j in range(quant_config['num_calib_samples']):
                batch_input = compressed_inputs[j:j + 1].to(dev)
                batch_output = q_block(batch_input, **kwargs)[0]
                compressed_inputs[j:j + 1] = batch_output.cpu().detach()
        q_blocks[i] = q_block.cpu()

        del q_block, fp_block, target_outputs
        cleanup_memory()

        test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
        print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")

    return model


def compress_model_recon(model, fp_model, dataloader, quant_config, dev="cuda"):
    """
    Use knowledge distillation to globally tune scales.
    """
    @torch.no_grad()
    def _compute_teacher_logits_cache(fp_model, dataloader, dev="cuda"):
        """
        Runs the fp_model over the entire indexable dataset and caches logits in CPU memory.

        Returns:
            teacher_logits_cache: dict[int, torch.Tensor] mapping sample index -> logits (CPU).
        """
        fp_model.eval()
        fp_model = fp_model.to(dev)

        teacher_logits_cache = {}
        data_indices = list(range(len(dataloader)))
        dataloader = dataloader.to(device=dev, non_blocking=True)
        dataloader = [dataloader[idx].unsqueeze(0) for idx in data_indices]

        pbar = tqdm(data_indices, desc="Precomputing teacher logits (FP model)")

        for idx in pbar:
            batch = dataloader[idx]
            outputs = fp_model(batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            teacher_logits_cache[idx] = logits.detach().cpu()

        # Move teacher back to CPU if you don’t need it further on GPU
        del fp_model
        cleanup_memory(verbose=True)

        return teacher_logits_cache

    def kl_loss_fn(student_logits, teacher_logits, mask, temperature: float = 1.0) -> torch.Tensor:
        """
        Standard Forward KL (FKL): KL(Teacher || Student)
        
        Description:
            - The standard objective for Knowledge Distillation.
            - Has a 'Mean-seeking' property, forcing the student to cover the entire teacher distribution.
            - Can lead to overestimation of low-probability regions (tail), potentially causing hallucinations in LLMs.
        
        Reference:
            Hinton et al. (2015). Distilling the Knowledge in a Neural Network.
        """
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)

        inf_mask = torch.isinf(student_logits)
        prod = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod, dim=-1).view(-1)

        # Minimize -x (which is CE)
        loss = -torch.sum(x * mask.view(-1), dim=0) / (torch.sum(mask.view(-1), dim=0) + 1e-8)
        return (temperature**2) * loss

    # set seed
    set_seed(quant_config['seed'])
    # load tokenizer
    tokenizer = load_tokenizer(quant_config['model_id'])

    model.cpu()

    # get fp model logits
    fp_model.eval()
    teacher_logits_cache = _compute_teacher_logits_cache(
        fp_model=fp_model,
        dataloader=dataloader,
    )

    # Identify Pad Token for Masking
    pad_token_id = -100
    if hasattr(model, "config"):
        model.config.use_cache = False  # Disable KV cache for training
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id

    model.train()
    # Enable Gradient Checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
    model.cuda()

    params_to_tune = []
    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.do_train = True
            for name, param in module.named_parameters():
                if 'scale' in name:
                    param.requires_grad = True
                    params_to_tune.append(param)

    print(f"Total number of scale parameters to tune: {len(params_to_tune)}")
    if not params_to_tune:
        print("No scales found to tune. Returning original model.")
        model.eval()
        return model

    optimizer = AdamW(params_to_tune, lr=quant_config['model_kd_lr'])
    epochs = quant_config["model_kd_epochs"]
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Prepare data indices and pre-load if needed
    data_indices = list(range(len(dataloader)))
    dataloader = dataloader.to(device=dev, non_blocking=True)
    dataloader = [dataloader[idx].unsqueeze(0) for idx in data_indices]

    # -------------------------------------------
    # 3) KD-tuning loop (student model)
    # -------------------------------------------
    with torch.enable_grad():
        step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            random.shuffle(data_indices)
            total_train_loss = torch.zeros(1, device=dev)

            for idx in data_indices:
                batch = dataloader[idx]

                # Mask Generation
                if pad_token_id != -100:
                    mask = (batch != pad_token_id).int().to(dev)
                else:
                    mask = torch.ones_like(batch).int().to(dev)

                # KD Loss
                student_outputs = model(batch)
                student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs
                teacher_logits = teacher_logits_cache[idx].to(dev, non_blocking=True)

                # Pass logits + mask to KD functions
                loss = kl_loss_fn(student_logits, teacher_logits, mask)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                total_train_loss += loss.detach()

            avg_train = total_train_loss / len(dataloader)
            print(f"Epoch {epoch} - Loss: {avg_train.item():.4f}")

    # -------------------------------------------
    # 4) Cleanup
    # -------------------------------------------
    del params_to_tune, optimizer, scheduler, dataloader, teacher_logits_cache
    cleanup_memory(verbose=True)

    for module in model.modules():
        if isinstance(module, NanoQuantLinear):
            module.do_train = False

    model.eval()
    return model
