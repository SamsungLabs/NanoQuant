// Copyright (c) 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/extension.h>

torch::Tensor nanoquant_dyn_forward(
    torch::Tensor x,
    torch::Tensor scale_g,
    torch::Tensor Ubits,
    std::optional<torch::Tensor> scale_l,
    torch::Tensor Vbits,
    torch::Tensor scale_h,
    int64_t num_thread_stage1 = 128, int64_t num_thread_stage2 = 128,
    int64_t num_row_per_warp_stage1 = 2, int64_t num_row_per_warp_stage2 = 2,
    int64_t num_acc_stage1 = 4, int64_t num_acc_stage2 = 4,
    int64_t num_pipeline_stage1 = 2, int64_t num_pipeline_stage2 = 2);
