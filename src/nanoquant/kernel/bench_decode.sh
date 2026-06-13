# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Configurable arguments
MODEL_NAME="meta-llama/Llama-3.2-1B"
MAX_NEW_TOKENS="32,64,128,256,512,1024"
PROMPT_TOKENS=128
BATCH_SIZE=1
ITERS=5
DTYPE="bfloat16"
PPL_TASK=""   # empty string
PROFILE_ENERGY=true
PROFILE_GPU_MEMORY=true

# Checkpoints for quantized models
CHECKPOINTS=(
    "../../Llama-3.2-1B-NQ-1bit.pt"
)

# Kernels to benchmark (gemv)
KERNELS=("gemv" "gemm" "gemlite")

# Loop over each checkpoint and each kernel
for ckpt in "${CHECKPOINTS[@]}"; do
    echo -e "\n\n[CHECKPOINT] $ckpt"

    for kernel in "${KERNELS[@]}"; do
        case "$kernel" in gemv|gemlite|gemm)
                echo -e "\n\n[TEST] $kernel kernel"
                CUDA_VISIBLE_DEVICES=0 python -u -m nanoquant.kernel.test_decode \
                    --model_name "$MODEL_NAME" \
                    --qmodel_ckpt "$ckpt" \
                    --use_quant_kernels True \
                    --quant_kernel_type "$kernel" \
                    --max_new_tokens $MAX_NEW_TOKENS \
                    --prompt_tokens $PROMPT_TOKENS \
                    --batch_size $BATCH_SIZE \
                    --iters $ITERS \
                    --dtype $DTYPE \
                    --ppl_task "$PPL_TASK" \
                    --profile_energy \
                    --profile_gpu_memory
                ;;
        esac
    done
done

# Run full precision (fp16) test once
echo -e "\n\n[TEST] full precision (no quant kernel)"
CUDA_VISIBLE_DEVICES=0 python -u -m nanoquant.kernel.test_decode \
    --model_name "$MODEL_NAME" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --prompt_tokens $PROMPT_TOKENS \
    --batch_size $BATCH_SIZE \
    --iters $ITERS \
    --dtype $DTYPE \
    --ppl_task "$PPL_TASK" \
    --profile_energy \
    --profile_gpu_memory