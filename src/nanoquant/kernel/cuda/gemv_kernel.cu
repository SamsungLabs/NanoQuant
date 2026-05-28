// Copyright (c) 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <type_traits>
#include <optional>

#include "cuda_utils.cuh"
#include "meta_utils.h"
#include "gemv.h"

// ========================================================================
// === Type Traits for Half/BFloat16 Abstraction ===
// ========================================================================

template<typename T> struct ScalarTraits;

template<> struct ScalarTraits<half> {
    using T2 = half2;
    static __device__ __forceinline__ T2 uint32_to_T2(uint32_t x) { return uint32_to_half2(x); }
    static __device__ __forceinline__ T2 float_to_T2(float x) { return __float2half2_rn(x); }
    static __device__ __forceinline__ half warp_reduce(T2 x) { return warp_reduce_sum_half2_ret_half(x); }
    static __device__ __forceinline__ float T_to_float(half x) { return __half2float(x); }
};

template<> struct ScalarTraits<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    static __device__ __forceinline__ T2 uint32_to_T2(uint32_t x) { return uint32_to_bf162(x); }
    static __device__ __forceinline__ T2 float_to_T2(float x) { return __float2bfloat162_rn(x); }
    static __device__ __forceinline__ __nv_bfloat16 warp_reduce(T2 x) { return warp_reduce_sum_bf162_ret_bf16(x); }
    static __device__ __forceinline__ float T_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
};

// ========================================================================
// === Templated Implementation Kernels ===
// ========================================================================

template <typename T, unsigned NUM_THREAD, unsigned NUM_ROW_PER_WARP = 2, unsigned NUM_ACC = 4, unsigned NUM_PIPELINE_STAGE = 2>
__forceinline__ __device__ void nanoquant_stage1_impl(
    const T* __restrict__ x,
    const T* __restrict__ scale_g,
    const uint32_t* __restrict__ Ubits,
    const T* __restrict__ scale_l,
    T* __restrict__ interm,
    int N, int R) {
    
    using T2 = typename ScalarTraits<T>::T2;
    static_assert(NUM_THREAD % WARP_SIZE == 0);
    constexpr unsigned NUM_WARP = NUM_THREAD / WARP_SIZE;
    constexpr unsigned NUM_ROW_PER_BLOCK = NUM_ROW_PER_WARP * NUM_WARP;

    auto tid = threadIdx.x % WARP_SIZE;
    auto wid = threadIdx.x / WARP_SIZE;
    const uint4* __restrict__ x8s = reinterpret_cast<const uint4*>(x);
    const uint4* __restrict__ sg8s = reinterpret_cast<const uint4*>(scale_g);

    struct RowAccum { T2 plane[NUM_ACC]; };
    struct StageTile {
        uint4 x;
        uint4 scale;
        uint32_t bits[NUM_ROW_PER_WARP];
    };
    struct RowContext { int index; RowAccum acc; };

    RowContext rows[NUM_ROW_PER_WARP] = {}; 
    StageTile pipe[NUM_PIPELINE_STAGE];

    const int N8 = N >> 3;
    const int wordsN = N >> 5;

    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        rows[irow].index = blockIdx.x * NUM_ROW_PER_BLOCK + wid + irow * NUM_WARP;
    }

    // Precompute per-row base offsets
    uint32_t row_base[NUM_ROW_PER_WARP];
    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        row_base[irow] = rows[irow].index * wordsN;
    }

    auto load_to_reg = [&](int ireg, int idx8) {
        StageTile& stage = pipe[ireg];
        stage.x = x8s[idx8];
        stage.scale = sg8s[idx8];

        const int word_idx = idx8 >> 2;
        #pragma unroll
        for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
            stage.bits[irow] = Ubits[row_base[irow] + word_idx];
        }
    };

    const int MASK = NUM_ACC - 1;
    auto fma = [=](const uint4& scale, const uint4& x_val, T2 acc[NUM_ACC]) {
        // Crucial: All 4 elements must be accumulated using MASK to avoid data loss
        acc[0 & MASK] = __hfma2(ScalarTraits<T>::uint32_to_T2(scale.x), ScalarTraits<T>::uint32_to_T2(x_val.x), acc[0 & MASK]);
        acc[1 & MASK] = __hfma2(ScalarTraits<T>::uint32_to_T2(scale.y), ScalarTraits<T>::uint32_to_T2(x_val.y), acc[1 & MASK]);
        acc[2 & MASK] = __hfma2(ScalarTraits<T>::uint32_to_T2(scale.z), ScalarTraits<T>::uint32_to_T2(x_val.z), acc[2 & MASK]);
        acc[3 & MASK] = __hfma2(ScalarTraits<T>::uint32_to_T2(scale.w), ScalarTraits<T>::uint32_to_T2(x_val.w), acc[3 & MASK]);
    };

    auto calc_main = [&](int ireg) {
        StageTile& stage = pipe[ireg];
        #pragma unroll
        for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
            // Apply sign based on bit pattern for this specific row
            uint4 tmp = apply_sign(stage.x, stage.bits[irow], (tid << 2) & 15);
            fma(stage.scale, tmp, rows[irow].acc.plane);
        }
    };

    // ---------- PIPELINING ----------
    int idx_load = tid;
    int idx_calc = tid;

    #pragma unroll
    for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
        if (idx_load < N8) { load_to_reg(istg, idx_load); }
        idx_load += WARP_SIZE;
    }

    const int bound_mainloop = N8 - (NUM_PIPELINE_STAGE * WARP_SIZE);
    for (; idx_calc < bound_mainloop;) {
        #pragma unroll
        for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
            calc_main(istg);
            idx_calc += WARP_SIZE;
        }
        #pragma unroll
        for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
            if (idx_load < N8) { load_to_reg(istg, idx_load); }
            idx_load += WARP_SIZE;
        }
    }

    #pragma unroll
    for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
        if (idx_calc < N8) { calc_main(istg); idx_calc += WARP_SIZE; }
        if (idx_load < N8) { load_to_reg(istg, idx_load); idx_load += WARP_SIZE; }
    }

    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        RowContext& row = rows[irow];
        T2 r = ScalarTraits<T>::float_to_T2(0.0f);
        #pragma unroll
        for (int k = 0; k < (int)NUM_ACC; ++k) r = __hadd2(r, row.acc.plane[k]);

        T warp_sum_h = ScalarTraits<T>::warp_reduce(r);
        if (tid == 0) {
            T res = warp_sum_h;
            if (scale_l != nullptr) res = __hmul(res, scale_l[row.index]);
            interm[row.index] = res;
        }
    }
}

template <typename T, unsigned NUM_THREAD, unsigned NUM_ROW_PER_WARP = 2, unsigned NUM_ACC = 4, unsigned NUM_PIPELINE_STAGE = 2>
__forceinline__ __device__ void nanoquant_stage2_impl(
    const T* __restrict__ interm,
    const uint32_t* __restrict__ Vbits,
    const T* __restrict__ scale_h,
    T* __restrict__ y,
    int R, int M) {
    
    using T2 = typename ScalarTraits<T>::T2;
    static_assert(NUM_THREAD % WARP_SIZE == 0);
    constexpr unsigned NUM_WARP = NUM_THREAD / WARP_SIZE;
    constexpr unsigned NUM_ROW_PER_BLOCK = NUM_ROW_PER_WARP * NUM_WARP;

    auto tid = threadIdx.x % WARP_SIZE;
    auto wid = threadIdx.x / WARP_SIZE;
    const uint4* __restrict__ im8s = reinterpret_cast<const uint4*>(interm);

    struct RowAccum { T2 plane[NUM_ACC]; };
    struct StageTile {
        uint4 im;
        uint32_t bits[NUM_ROW_PER_WARP];
    };
    struct RowContext { int index; RowAccum acc; };

    RowContext rows[NUM_ROW_PER_WARP] = {};
    StageTile pipe[NUM_PIPELINE_STAGE];

    const int R8 = R >> 3;
    const int wordsR = R >> 5;

    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        rows[irow].index = blockIdx.x * NUM_ROW_PER_BLOCK + wid + irow * NUM_WARP;
    }

    // Precompute per-row base offsets
    uint32_t row_base[NUM_ROW_PER_WARP];
    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        row_base[irow] = rows[irow].index * wordsR;
    }

    auto load_to_reg = [&](int ireg, int idx8) {
        StageTile& stage = pipe[ireg];
        stage.im = im8s[idx8];

        const int word_idx = idx8 >> 2;
        #pragma unroll
        for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
            stage.bits[irow] = Vbits[row_base[irow] + word_idx];
        }
    };

    const int MASK = NUM_ACC - 1;
    auto fma = [=](const uint4& im, const uint32_t bits, T2 acc[NUM_ACC]) {
        uint4 im_sgn = apply_sign(im, bits, (tid << 2) & 15);
        acc[0 & MASK] = __hadd2(ScalarTraits<T>::uint32_to_T2(im_sgn.x), acc[0 & MASK]);
        acc[1 & MASK] = __hadd2(ScalarTraits<T>::uint32_to_T2(im_sgn.y), acc[1 & MASK]);
        acc[2 & MASK] = __hadd2(ScalarTraits<T>::uint32_to_T2(im_sgn.z), acc[2 & MASK]);
        acc[3 & MASK] = __hadd2(ScalarTraits<T>::uint32_to_T2(im_sgn.w), acc[3 & MASK]);
    };

    auto calc_main = [&](int ireg) {
        StageTile& stage = pipe[ireg];
        #pragma unroll
        for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
            fma(stage.im, stage.bits[irow], rows[irow].acc.plane);
        }
    };

    // ---------- PIPELINING ----------
    int idx_load = tid;
    int idx_calc = tid;

    #pragma unroll
    for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
        if (idx_load < R8) { load_to_reg(istg, idx_load); }
        idx_load += WARP_SIZE;
    }

    const int bound_mainloop = R8 - (NUM_PIPELINE_STAGE * WARP_SIZE);
    for (; idx_calc < bound_mainloop;) {
        #pragma unroll
        for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
            calc_main(istg);
            idx_calc += WARP_SIZE;
        }
        #pragma unroll
        for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
            if (idx_load < R8) { load_to_reg(istg, idx_load); }
            idx_load += WARP_SIZE;
        }
    }

    #pragma unroll
    for (int istg = 0; istg < NUM_PIPELINE_STAGE; ++istg) {
        if (idx_calc < R8) { calc_main(istg); idx_calc += WARP_SIZE; }
        if (idx_load < R8) { load_to_reg(istg, idx_load); idx_load += WARP_SIZE; }
    }

    #pragma unroll
    for (unsigned irow = 0; irow < NUM_ROW_PER_WARP; ++irow) {
        RowContext& row = rows[irow];
        T2 r = ScalarTraits<T>::float_to_T2(0.0f);
        #pragma unroll
        for (int k = 0; k < (int)NUM_ACC; ++k) r = __hadd2(r, row.acc.plane[k]);

        T warp_sum_h = ScalarTraits<T>::warp_reduce(r);
        if (tid == 0) {
            y[row.index] = clamp_inf_for_half<T>(ScalarTraits<T>::T_to_float(__hmul(warp_sum_h, scale_h[row.index])));
        }
    }
}

// ========================================================================
// === Global Kernel Launchers ===
// ========================================================================

template <typename T, unsigned NUM_THREAD, unsigned NUM_ROW_PER_WARP, unsigned NUM_ACC, unsigned NUM_PIPELINE_STAGE>
__launch_bounds__(NUM_THREAD)
__global__ void nanoquant_stage1_dyn_kernel(
    const T* __restrict__ x, const T* __restrict__ scale_g, const uint32_t* __restrict__ Ubits,
    const T* __restrict__ scale_l, T* __restrict__ interm, int N, int R) {
    const int s = blockIdx.y;
    nanoquant_stage1_impl<T, NUM_THREAD, NUM_ROW_PER_WARP, NUM_ACC, NUM_PIPELINE_STAGE>(
        x + s * N, scale_g, Ubits, scale_l, interm + s * R, N, R);
}

template <typename T, unsigned NUM_THREAD, unsigned NUM_ROW_PER_WARP, unsigned NUM_ACC, unsigned NUM_PIPELINE_STAGE>
__launch_bounds__(NUM_THREAD)
__global__ void nanoquant_stage2_dyn_kernel(
    const T* __restrict__ interm, const uint32_t* __restrict__ Vbits, const T* __restrict__ scale_h,
    T* __restrict__ y, int R, int M) {
    const int s = blockIdx.y;
    nanoquant_stage2_impl<T, NUM_THREAD, NUM_ROW_PER_WARP, NUM_ACC, NUM_PIPELINE_STAGE>(
        interm + s * R, Vbits, scale_h, y + s * M, R, M);
}


// ========================================================================
// === C++ Wrapper Function (Dispatch Logic) ===
// ========================================================================

torch::Tensor nanoquant_dyn_forward(
    torch::Tensor x, torch::Tensor scale_g, torch::Tensor Ubits, std::optional<torch::Tensor> scale_l,
    torch::Tensor Vbits, torch::Tensor scale_h,
    int64_t num_thread_stage1, int64_t num_thread_stage2,
    int64_t num_row_per_warp_stage1, int64_t num_row_per_warp_stage2,
    int64_t num_acc_stage1, int64_t num_acc_stage2,
    int64_t num_pipeline_stage1, int64_t num_pipeline_stage2) {

    // 1. Dispatch based on ScalarType
    auto scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16, "Support only Half or BFloat16");

    if (scalar_type == at::kHalf) {
        CHECK_CUDA_CONT_F16(x);
        CHECK_CUDA_CONT_F16(scale_g);
        CHECK_CUDA_CONT_F16(scale_h);
    } else {
        CHECK_CUDA_CONT_BF16(x);
        CHECK_CUDA_CONT_BF16(scale_g);
        CHECK_CUDA_CONT_BF16(scale_h);
    }
    CHECK_CUDA_CONT_INT32(Ubits);
    CHECK_CUDA_CONT_INT32(Vbits);

    const void* scale_l_ptr = nullptr;
    if (scale_l.has_value() && scale_l->defined()) {
        if (scalar_type == at::kHalf) CHECK_CUDA_CONT_F16((*scale_l));
        else CHECK_CUDA_CONT_BF16((*scale_l));
        TORCH_CHECK(scale_l->size(0) == Ubits.size(0), "scale_l size mismatch");
        scale_l_ptr = scale_l->data_ptr();
    }

    const int seqlen = x.dim() == 1 ? 1 : static_cast<int>(x.size(0));
    const int N = static_cast<int>(x.size(-1));
    const int R = static_cast<int>(Ubits.size(0));
    const int M = static_cast<int>(scale_h.size(0));

    auto interm = torch::empty({(long long)seqlen, (long long)R}, x.options());
    auto y = torch::empty({(long long)seqlen, (long long)M}, x.options());

    at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    parameter_sweep<unsigned, 64u, 128u, 256u> sweep_num_thread;
    parameter_sweep<unsigned, 1, 2, 4, 8> sweep_row_per_warp;
    parameter_sweep<unsigned, 1, 2> sweep_num_acc;
    parameter_sweep<unsigned, 1, 2, 4, 8, 16> sweep_pipeline_stage;

    // Dispatch helper for kernels
    auto dispatch_stage1 = [&](auto n_th, auto n_rpw, auto n_acc, auto n_pipe) {
        constexpr unsigned th = n_th.value;
        constexpr unsigned rpw = n_rpw.value;
        constexpr unsigned acc = n_acc.value;
        constexpr unsigned pipe = n_pipe.value;
        dim3 grid(DIV_UP(R, (th / 32 * rpw)), seqlen);

        if (scalar_type == at::kHalf) {
            nanoquant_stage1_dyn_kernel<half, th, rpw, acc, pipe><<<grid, th, 0, stream.stream()>>>(
                reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(scale_g.data_ptr<at::Half>()),
                reinterpret_cast<const uint32_t*>(Ubits.data_ptr<int32_t>()),
                reinterpret_cast<const half*>(scale_l_ptr),
                reinterpret_cast<half*>(interm.data_ptr<at::Half>()), N, R);
        } else {
            nanoquant_stage1_dyn_kernel<__nv_bfloat16, th, rpw, acc, pipe><<<grid, th, 0, stream.stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(scale_g.data_ptr<at::BFloat16>()),
                reinterpret_cast<const uint32_t*>(Ubits.data_ptr<int32_t>()),
                reinterpret_cast<const __nv_bfloat16*>(scale_l_ptr),
                reinterpret_cast<__nv_bfloat16*>(interm.data_ptr<at::BFloat16>()), N, R);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    auto dispatch_stage2 = [&](auto n_th, auto n_rpw, auto n_acc, auto n_pipe) {
        constexpr unsigned th = n_th.value;
        constexpr unsigned rpw = n_rpw.value;
        constexpr unsigned acc = n_acc.value;
        constexpr unsigned pipe = n_pipe.value;
        dim3 grid(DIV_UP(M, (th / 32 * rpw)), seqlen);

        if (scalar_type == at::kHalf) {
            nanoquant_stage2_dyn_kernel<half, th, rpw, acc, pipe><<<grid, th, 0, stream.stream()>>>(
                reinterpret_cast<const half*>(interm.data_ptr<at::Half>()),
                reinterpret_cast<const uint32_t*>(Vbits.data_ptr<int32_t>()),
                reinterpret_cast<const half*>(scale_h.data_ptr<at::Half>()),
                reinterpret_cast<half*>(y.data_ptr<at::Half>()), R, M);
        } else {
            nanoquant_stage2_dyn_kernel<__nv_bfloat16, th, rpw, acc, pipe><<<grid, th, 0, stream.stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(interm.data_ptr<at::BFloat16>()),
                reinterpret_cast<const uint32_t*>(Vbits.data_ptr<int32_t>()),
                reinterpret_cast<const __nv_bfloat16*>(scale_h.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), R, M);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    // Apply sweeps for stage 1
    sweep_num_thread("n_th1", num_thread_stage1, [&](auto n_th) {
        sweep_row_per_warp("n_rpw1", num_row_per_warp_stage1, [&](auto n_rpw) {
            sweep_num_acc("n_acc1", num_acc_stage1, [&](auto n_acc) {
                sweep_pipeline_stage("n_pipe1", num_pipeline_stage1, [&](auto n_pipe) {
                    dispatch_stage1(n_th, n_rpw, n_acc, n_pipe);
                });
            });
        });
    });

    // Apply sweeps for stage 2
    sweep_num_thread("n_th2", num_thread_stage2, [&](auto n_th) {
        sweep_row_per_warp("n_rpw2", num_row_per_warp_stage2, [&](auto n_rpw) {
            sweep_num_acc("n_acc2", num_acc_stage2, [&](auto n_acc) {
                sweep_pipeline_stage("n_pipe2", num_pipeline_stage2, [&](auto n_pipe) {
                    dispatch_stage2(n_th, n_rpw, n_acc, n_pipe);
                });
            });
        });
    });

    if (x.dim() == 1) y.squeeze_(0);
    return y;
}
