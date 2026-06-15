/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modified by Samsung Electronics Co., Ltd.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

template <typename T>
int marlin_cuda_template(
    const void* A, const void* B, void* C, void* s_in, void* s_out,
    int prob_m, int prob_n, int prob_k, void* workspace,
    int dev, cudaStream_t stream, int thread_k, int thread_n, int sms, int max_par);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;
const int ERR_MAX_PAR = 3;
const int MARLIN_MIN_MAX_PAR = 32;

// Common check logic
void marlin_onebit_check_and_alloc(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_out,
    const torch::Tensor& workspace, int max_par) {

    int prob_m = A.size(0);
    int prob_n = B.size(1) * 2;
    int prob_k = A.size(1);

    TORCH_CHECK(prob_k == B.size(0) * 16, "k dimension mismatch");
    TORCH_CHECK(!s_in.has_value() || s_in->size(0) == prob_k, "s_in size mismatch");
    TORCH_CHECK(!s_out.has_value() || s_out->size(0) == prob_n, "s_out size mismatch");
    TORCH_CHECK(max_par >= MARLIN_MIN_MAX_PAR, "max_par must be at least ", MARLIN_MIN_MAX_PAR);
    TORCH_CHECK(workspace.numel() >= prob_n / 128 * max_par, "workspace too small");

    // Set to follow A's dtype as is (Half or BFloat16)
    C = torch::empty({prob_m, prob_n}, A.options());
}

// Integrated CUDA entry point
torch::Tensor marlin_onebit_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace) {

    int max_par = MARLIN_MIN_MAX_PAR;
    torch::Tensor C;
    marlin_onebit_check_and_alloc(A, B, C, s_in, s_out, workspace, max_par);

    int prob_m = A.size(0);
    int prob_n = C.size(1);
    int prob_k = A.size(1);
    int dev = A.get_device();

    auto s_in_ptr = s_in.has_value() ? s_in->data_ptr() : nullptr;
    auto s_out_ptr = s_out.has_value() ? s_out->data_ptr() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
    cudaMemsetAsync(workspace.data_ptr(), 0, workspace.nbytes(), stream);

    int err;
    if (A.scalar_type() == torch::kHalf) {
        err = marlin_cuda_template<half>(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), s_in_ptr, s_out_ptr,
            prob_m, prob_n, prob_k, workspace.data_ptr(), dev, stream, -1, -1, -1, max_par);
    } else if (A.scalar_type() == torch::kBFloat16) {
        err = marlin_cuda_template<__nv_bfloat16>(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), s_in_ptr, s_out_ptr,
            prob_m, prob_n, prob_k, workspace.data_ptr(), dev, stream, -1, -1, -1, max_par);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: Marlin only supports Half and BFloat16");
    }

    TORCH_CHECK(err != ERR_PROB_SHAPE, "Problem shape incompatible");
    TORCH_CHECK(err != ERR_KERN_SHAPE, "No kernel implementation found");
    TORCH_CHECK(err != ERR_MAX_PAR, "max_par must be at least ", MARLIN_MIN_MAX_PAR);
    
    return C;
}

torch::Tensor marlin_onebit_meta(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace) {
    int max_par = MARLIN_MIN_MAX_PAR;
    torch::Tensor C;
    marlin_onebit_check_and_alloc(A, B, C, s_in, s_out, workspace, max_par);
    return C;
}

torch::Tensor marlin_nanoquant_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_first,
    const torch::Tensor& B_second,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_imm,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace) {
    torch::Tensor imm = marlin_onebit_cuda(A, B_first, s_in, s_imm, workspace);
    return marlin_onebit_cuda(imm, B_second, std::nullopt, s_out, workspace);
}

torch::Tensor marlin_nanoquant_meta(
    const torch::Tensor& A,
    const torch::Tensor& B_first,
    const torch::Tensor& B_second,
    const std::optional<torch::Tensor>& s_in,
    const std::optional<torch::Tensor>& s_imm,
    const std::optional<torch::Tensor>& s_out,
    torch::Tensor& workspace) {
    torch::Tensor imm = marlin_onebit_meta(A, B_first, s_in, s_imm, workspace);
    return marlin_onebit_meta(imm, B_second, std::nullopt, s_out, workspace);
}
