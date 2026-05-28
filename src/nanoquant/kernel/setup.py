# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _find_compiler():
    """Find an available C++ compiler across platforms."""
    for candidate in ("g++", "gcc", "clang++", "cl"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def get_arch_list():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        print(f"\n[INFO] Local GPU Detected: {major}.{minor}")
        print(f"[INFO] Compiling ONLY for compute_{arch}, sm_{arch} (Fastest Build)\n")
        return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]

    print("\n[WARN] No GPU detected. Compiling for default architectures (sm_80, sm_90).\n")
    return [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def get_nvcc_args():
    nvcc = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-t", "32",
    ] + get_arch_list()

    ccbin = _find_compiler()
    if ccbin:
        nvcc.append(f"-ccbin={ccbin}")
    else:
        print("[WARN] No C++ compiler found in PATH. NVCC may fail to locate host compiler.")

    return nvcc


setup(
    name="binary_kernels",
    ext_modules=[
        CUDAExtension(
            name="binary_kernels",
            sources=[
                "pybind.cpp",
                "cuda/gemm.cpp",
                "cuda/gemv_kernel.cu",
                "cuda/gemm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": get_nvcc_args(),
            },
        ),
    ],
    options={"build_ext": {"parallel": 32}},
    cmdclass={"build_ext": BuildExtension},
)
