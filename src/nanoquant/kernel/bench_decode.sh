# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Configurable arguments. Override as env vars, for example:
#   ITERS=10 KERNELS="gemm gemlite" bash bench_decode.sh
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32,64,128,256,512,1024}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-5}"
DTYPE="${DTYPE:-bfloat16}"
PPL_TASK="${PPL_TASK:-}"
BENCH_PROMPT="${BENCH_PROMPT:-Follow the given instructions: }"
GPU_ID="${GPU_ID:-0}"
SEQLEN="${SEQLEN:-2048}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
STREAMING="${STREAMING:-0}"
PROFILE_ENERGY="${PROFILE_ENERGY:-0}"
PROFILE_GPU_MEMORY="${PROFILE_GPU_MEMORY:-0}"
ZEUS_LOG_FILE="${ZEUS_LOG_FILE:-}"
ZEUS_APPROX_INSTANT_ENERGY="${ZEUS_APPROX_INSTANT_ENERGY:-0}"
EXTRA_TEST_DECODE_ARGS="${EXTRA_TEST_DECODE_ARGS:-}"

# Space-separated lists. Keep checkpoint paths free of spaces.
CHECKPOINTS="${CHECKPOINTS:-../../Llama-3.2-1B-NQ-1bit.pt}"
KERNELS="${KERNELS:-gemv gemm gemlite}"
read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS"
read -r -a KERNEL_LIST <<< "$KERNELS"

enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

COMMON_ARGS=(
    --model_name "$MODEL_NAME"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --prompt_tokens "$PROMPT_TOKENS"
    --batch_size "$BATCH_SIZE"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --dtype "$DTYPE"
    --ppl_task "$PPL_TASK"
    --bench_prompt "$BENCH_PROMPT"
    --gpu_id "$GPU_ID"
    --seqlen "$SEQLEN"
)

if [[ -n "$OUTPUT_CSV" ]]; then
    COMMON_ARGS+=(--output_csv "$OUTPUT_CSV")
fi
if enabled "$STREAMING"; then
    COMMON_ARGS+=(--streaming)
fi

PROFILE_ARGS=()
if enabled "$PROFILE_ENERGY"; then
    PROFILE_ARGS+=(--profile_energy)
fi
if enabled "$PROFILE_GPU_MEMORY"; then
    PROFILE_ARGS+=(--profile_gpu_memory)
fi
if [[ -n "$ZEUS_LOG_FILE" ]]; then
    PROFILE_ARGS+=(--zeus_log_file "$ZEUS_LOG_FILE")
fi
if enabled "$ZEUS_APPROX_INSTANT_ENERGY"; then
    PROFILE_ARGS+=(--zeus_approx_instant_energy)
fi

EXTRA_ARGS=()
if [[ -n "$EXTRA_TEST_DECODE_ARGS" ]]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA_TEST_DECODE_ARGS"
fi

# Loop over each checkpoint and each kernel
for ckpt in "${CHECKPOINT_LIST[@]}"; do
    echo -e "\n\n[CHECKPOINT] $ckpt"

    for kernel in "${KERNEL_LIST[@]}"; do
        case "$kernel" in gemv|gemlite|gemm)
                echo -e "\n\n[TEST] $kernel kernel"
                python -u -m nanoquant.kernel.test_decode \
                    "${COMMON_ARGS[@]}" \
                    --qmodel_ckpt "$ckpt" \
                    --use_quant_kernels True \
                    --quant_kernel_type "$kernel" \
                    "${PROFILE_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}"
                ;;
            *)
                echo "[WARN] skipping unsupported kernel: $kernel"
                ;;
        esac
    done
done

# Run full precision test once
echo -e "\n\n[TEST] full precision (no quant kernel)"
python -u -m nanoquant.kernel.test_decode \
    "${COMMON_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
