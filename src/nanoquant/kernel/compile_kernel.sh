#!/usr/bin/env bash
# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Compile NanoQuant CUDA kernels
#
# Supports both CUDA 12.x and CUDA 13.x toolkits:
#   - CUDA 12.x: thrust headers are at <cuda>/include/thrust/ (no extra action needed)
#   - CUDA 13.x: thrust headers moved to <cuda>/include/cccl/thrust/ due to CCCL
#     packaging changes (CUDA C++ Core Libraries). We detect this and add the
#     cccl directory to CPATH so #include <thrust/complex.h> resolves correctly.
#
# Usage:
#   cd src/nanoquant/kernel
#   bash compile_kernel.sh

set -e

# Detect CCCL path for CUDA 13+ where thrust is under cccl/
find_cccl() {
    # 1. Check active conda environment
    for conda_base in "${CONDA_PREFIX}"; do
        if [ -n "$conda_base" ]; then
            cccl="${conda_base}/targets/x86_64-linux/include/cccl"
            if [ -d "$cccl" ]; then
                echo "$cccl"
                return
            fi
        fi
    done

    # 2. Check standard CUDA toolkit locations
    for cuda_home in "${CUDA_HOME}" "${CUDA_PATH}" "/usr/local/cuda"; do
        if [ -n "$cuda_home" ]; then
            cccl="${cuda_home}/targets/x86_64-linux/include/cccl"
            if [ -d "$cccl" ]; then
                echo "$cccl"
                return
            fi
        fi
    done

    # 3. Fallback: known environments with full thrust headers
    if [ -f "/usr/include/thrust/complex.h" ]; then
        echo "/usr/include"
        return
    fi
}

CCCL_INCLUDE=$(find_cccl)

if [ -n "$CCCL_INCLUDE" ]; then
    echo "[INFO] Found CCCL/thrust include path: $CCCL_INCLUDE"
    export CPATH="${CCCL_INCLUDE}:${CPATH}"
    export C_INCLUDE_PATH="${CCCL_INCLUDE}:${C_INCLUDE_PATH}"
    export CPLUS_INCLUDE_PATH="${CCCL_INCLUDE}:${CPLUS_INCLUDE_PATH}"
else
    echo "[INFO] CCCL include path not found. Assuming CUDA 12.x layout (thrust in standard include path)."
fi

rm -rf build
rm -rf *.so

# Build with torch
python setup.py build_ext --inplace

echo "[INFO] Kernel compilation complete."
