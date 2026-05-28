// Copyright (c) 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <torch/extension.h>

// --- Include Headers ---
#include "cuda/gemv.h"

// Meta implementations for all kernels
torch::Tensor nanoquant_meta_impl(
    torch::Tensor x,
    torch::Tensor scale_g,
    torch::Tensor Ubits,
    std::optional<torch::Tensor> scale_l,
    torch::Tensor Vbits,
    torch::Tensor scale_h,
    int64_t num_thread_stage1,
    int64_t num_thread_stage2,
    int64_t num_row_per_warp_stage1,
    int64_t num_row_per_warp_stage2,
    int64_t num_acc_stage1,
    int64_t num_acc_stage2,
    int64_t num_pipeline_stage1,
    int64_t num_pipeline_stage2) {
    int64_t M = scale_h.size(0);

    auto output_shape = x.sizes().vec();
    output_shape.back() = M;

    return torch::empty(output_shape, x.options());
}

torch::Tensor onebit_meta_impl(
    torch::Tensor x,
    torch::Tensor scale_g,
    torch::Tensor Wbits,
    torch::Tensor scale_h,
    int64_t num_thread,
    int64_t num_row_per_warp,
    int64_t num_acc) {
    int64_t M = scale_h.size(0);

    auto output_shape = x.sizes().vec();
    output_shape.back() = M;

    return torch::empty(output_shape, x.options());
}

torch::Tensor marlin_onebit_meta(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace);

torch::Tensor marlin_onebit_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace);

torch::Tensor marlin_nanoquant_meta(
    const torch::Tensor& A,
    const torch::Tensor& B_first,
    const torch::Tensor& B_second,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_imm,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace);

torch::Tensor marlin_nanoquant_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_first,
    const torch::Tensor& B_second,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_imm,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace);

// Register all kernels in a single library
TORCH_LIBRARY(binary_kernels, m) {
    // NanoQuant kernel
    m.def(
        "nanoquant_dyn_forward("
        "Tensor x, Tensor scale_g, Tensor Ubits, Tensor? scale_l, Tensor Vbits, Tensor scale_h, "
        "int num_thread_stage1=128, int num_thread_stage2=128, int num_row_per_warp_stage1=2, "
        "int num_row_per_warp_stage2=2, int num_acc_stage1=4, int num_acc_stage2=4, "
        "int num_pipeline_stage1=2, int num_pipeline_stage2=2) -> Tensor");

    // Marlin kernel
     m.def(
        "marlin_onebit_forward("
        "Tensor x, "
        "Tensor weight, "
        "Tensor? scale_g, Tensor? scale_h, "
        "Tensor(a!) workspace) -> Tensor");
    m.def(
        "marlin_nanoquant_forward("
        "Tensor x, "
        "Tensor weight_u, Tensor weight_v, "
        "Tensor? scale_g, Tensor? scale_l, Tensor? scale_h, "
        "Tensor(a!) workspace) -> Tensor");
}

// Implement all kernels for CUDA and Meta
TORCH_LIBRARY_IMPL(binary_kernels, CUDA, m) {
    // NanoQuant kernel
    m.impl("nanoquant_dyn_forward", &nanoquant_dyn_forward);

    // Marlin kernel
    m.impl("marlin_onebit_forward", TORCH_FN(marlin_onebit_cuda));
    m.impl("marlin_nanoquant_forward", TORCH_FN(marlin_nanoquant_cuda));
}

TORCH_LIBRARY_IMPL(binary_kernels, Meta, m) {
    // NanoQuant kernel
    m.impl("nanoquant_dyn_forward", &nanoquant_meta_impl);

    // Marlin kernel
    m.impl("marlin_onebit_forward", TORCH_FN(marlin_onebit_meta));
    m.impl("marlin_nanoquant_forward", TORCH_FN(marlin_nanoquant_meta));
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA Kernels: NanoQuant";
    // All function bindings are now registered through TORCH_LIBRARY
}
