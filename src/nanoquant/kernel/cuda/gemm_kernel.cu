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

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda_utils.cuh"

constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

constexpr int max_int(int a, int b) {
    return a > b ? a : b;
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) {
        return elems[i];
    }
    __device__ const T& operator[](int i) const {
        return elems[i];
    }
};

using I1 = Vec<int, 1>;
using FragC = Vec<float, 4>;

template <
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const int stages,
    const bool use_s_in,
    const bool use_s_out
>
struct MarlinSharedMemLayout {
    static_assert(thread_n_blocks % 4 == 0, "thread_n_blocks must be a multiple of 4");

    static constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
    static constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);

    static constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    static constexpr int true_b_sh_stride = 2 * thread_n_blocks;
    static constexpr int true_b_sh_stage = true_b_sh_stride * thread_k_blocks;

    static constexpr int s_in_sh_stride = 16 * thread_k_blocks / 8;
    static constexpr int s_in_sh_stage = use_s_in ? s_in_sh_stride : 0;
    static constexpr int s_out_sh_stride = 16 * thread_n_blocks / 8;
    static constexpr int s_out_sh_stage = use_s_out ? s_out_sh_stride : 0;

    static constexpr int a_offset_int4 = 0;
    static constexpr int b_offset_int4 = a_offset_int4 + stages * a_sh_stage;
    static constexpr int s_in_offset_int4 = b_offset_int4 + stages * true_b_sh_stage;
    static constexpr int s_out_offset_int4 = s_in_offset_int4 + stages * s_in_sh_stage;
    static constexpr int pipeline_int4 = s_out_offset_int4 + s_out_sh_stage;

    static constexpr int red_groups = threads / b_sh_stride;
    static constexpr int reduce_int4 = red_groups > 1 ? (8 * threads - b_sh_stride) : 0;

    static constexpr int active_threads = 32 * thread_n_blocks / 4;
    static constexpr int global_reduce_int4 = active_threads * thread_m_blocks * 4;

    static constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    static constexpr int max_writer_warp = thread_n_blocks / 4 - 1;
    static constexpr int max_c_sh_wr = (4 * c_sh_stride) * 7 + 3 + 32 * max_writer_warp;
    static constexpr int max_write_idx = max_c_sh_wr + 8 * 3 + (4 * c_sh_stride) * 8 + 4
                                        + (thread_m_blocks - 1) * 16 * (4 * c_sh_stride);
    static constexpr int write_result_int4 = (max_write_idx + 4) / 4;

    static constexpr int scratch_int4 =
        max_int(max_int(reduce_int4, global_reduce_int4), write_result_int4);
    static constexpr int smem_int4 = max_int(pipeline_int4, scratch_int4);
    static constexpr int bytes = smem_int4 * static_cast<int>(sizeof(int4));

    static constexpr bool covers_pipeline = smem_int4 >= pipeline_int4;
    static constexpr bool covers_reduce = smem_int4 >= reduce_int4;
    static constexpr bool covers_global_reduce = smem_int4 >= global_reduce_int4;
    static constexpr bool covers_write_result = smem_int4 >= write_result_int4;

    // s_out is copied into the pipeline area before thread_block_reduce() reuses sh[] as scratch.
    static constexpr bool reduce_before_s_out =
        !use_s_out || reduce_int4 <= s_out_offset_int4;

    static_assert(covers_pipeline, "Marlin shared memory must cover the fetch pipeline");
    static_assert(covers_reduce, "Marlin shared memory must cover thread_block_reduce scratch");
    static_assert(covers_global_reduce, "Marlin shared memory must cover global_reduce scratch");
    static_assert(covers_write_result, "Marlin shared memory must cover write_result scratch");
    static_assert(reduce_before_s_out, "Marlin reduction scratch overlaps pending s_out copy");
};

// --- Type Traits for Template Dispatch ---
template<typename T> struct MarlinTraits;

template<> struct MarlinTraits<half> {
    using Scalar = half;
    using Vec2 = half2;
    static constexpr unsigned ONE_BITS = 0x3c003c00;
    static __device__ __forceinline__ Vec2 make_vec2(float a, float b) { return __halves2half2(__float2half(a), __float2half(b)); }
    static __device__ __forceinline__ Vec2 from_scalars(Scalar a, Scalar b) { return __halves2half2(a, b); }
    static constexpr float MAX_FINITE = 65504.0f;
    // FP16 has a much smaller finite range than the FP32 accumulator, so clamp before narrowing.
    static __device__ __forceinline__ Scalar from_float(float v) { return __float2half(fminf(fmaxf(v, -MAX_FINITE), MAX_FINITE)); }
    static __device__ __forceinline__ float to_float(Scalar v) { return __half2float(v); }
    static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float get_low_float(Vec2 v) { return __low2float(v); }
    static __device__ __forceinline__ float get_high_float(Vec2 v) { return __high2float(v); }

    static __device__ __forceinline__ void mma(const Vec<half2, 4>& a_frag, const Vec<half2, 2>& b_frag, FragC& c_frag) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
        const uint32_t* b = reinterpret_cast<const uint32_t*>(&b_frag);
        float* c = reinterpret_cast<float*>(&c_frag);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

template<> struct MarlinTraits<__nv_bfloat16> {
    using Scalar = __nv_bfloat16;
    using Vec2 = __nv_bfloat162;
    static constexpr unsigned ONE_BITS = 0x3f803f80;
    static __device__ __forceinline__ Vec2 make_vec2(float a, float b) { return __halves2bfloat162(__float2bfloat16(a), __float2bfloat16(b)); }
    static __device__ __forceinline__ Vec2 from_scalars(Scalar a, Scalar b) { return __halves2bfloat162(a, b); }
    // BF16 keeps the FP32 exponent range, so the FP16 overflow clamp is not needed here.
    static __device__ __forceinline__ Scalar from_float(float v) { return __float2bfloat16(v); }
    static __device__ __forceinline__ float to_float(Scalar v) { return __bfloat162float(v); }
    static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float get_low_float(Vec2 v) { return __bfloat162float(__low2bfloat16(v)); }
    static __device__ __forceinline__ float get_high_float(Vec2 v) { return __bfloat162float(__high2bfloat16(v)); }

    static __device__ __forceinline__ void mma(const Vec<__nv_bfloat162, 4>& a_frag, const Vec<__nv_bfloat162, 2>& b_frag, FragC& c_frag) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
        const uint32_t* b = reinterpret_cast<const uint32_t*>(&b_frag);
        float* c = reinterpret_cast<float*>(&c_frag);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

// --- Common PTX wrappers ---
// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "    .reg .pred p;\n"
        "    setp.ne.b32 p, %0, 0;\n"
        "    @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

// Asynchronous global->shared copy with a cache hint indicating that the values may be evicted immediately; used for
// quantized weights B, which are only accessed precisely once and should thus not pollute the L2 cache which we need
// for inputs A and outputs C. 
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0; \n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" ::"r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(void* frag_ptr, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(frag_ptr);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

//
template <typename T>
__device__ inline Vec<typename MarlinTraits<T>::Vec2, 2> dequant_and_scale(unsigned q, const Vec<typename MarlinTraits<T>::Vec2, 2>& frag_s, int idx) {
    Vec<typename MarlinTraits<T>::Vec2, 2> frag_b;
    const unsigned SIGN_MASK = 0x80008000;
    *reinterpret_cast<unsigned*>(&frag_b[0]) = *reinterpret_cast<const unsigned*>(&frag_s[0]) ^ ((q << (idx * 2 + 0)) & SIGN_MASK);
    *reinterpret_cast<unsigned*>(&frag_b[1]) = *reinterpret_cast<const unsigned*>(&frag_s[1]) ^ ((q << (idx * 2 + 1)) & SIGN_MASK);
    return frag_b;
}

template <typename T>
__device__ inline Vec<typename MarlinTraits<T>::Vec2, 2> dequant_unit(unsigned q, int idx) {
    Vec<typename MarlinTraits<T>::Vec2, 2> frag_b;
    const unsigned SIGN_MASK = 0x80008000;
    const unsigned ONE_BITS = MarlinTraits<T>::ONE_BITS;
    *reinterpret_cast<unsigned*>(&frag_b[0]) = ONE_BITS ^ ((q << (idx * 2 + 0)) & SIGN_MASK);
    *reinterpret_cast<unsigned*>(&frag_b[1]) = ONE_BITS ^ ((q << (idx * 2 + 1)) & SIGN_MASK);
    return frag_b;
}

// --- Synchronization ---
__device__ inline void barrier_acquire(int* lock, int count) {
    if (threadIdx.x == 0) {
        int state = -1;
        do {
            asm volatile(
                "ld.global.acquire.gpu.b32 %0, [%1];\n"
                : "=r"(state) : "l"(lock)
            );
        } while (state != count);
    }
    __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) {
            lock[0] = 0;
            return;
        }
        int val = 1;
        // Make sure that all writes since acquiring this barrier are visible globally, while releasing the barrier.
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(lock), "r"(val));
    }
}

template <
    typename T,
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const int stages,
    const bool use_s_in,
    const bool use_s_out
>
__device__ inline void Marlin_impl(
    const int4* __restrict__ A,
    const int4* __restrict__ B,
    int4* __restrict__ C,
    const int4* __restrict__ s_in,
    const int4* __restrict__ s_out,
    int prob_m,
    int prob_n,
    int prob_k,
    int* locks
) {
    using Traits = MarlinTraits<T>;
    using TVec2 = typename Traits::Vec2;
    using Smem = MarlinSharedMemLayout<
        threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, use_s_in, use_s_out>;

    static_assert(
        Smem::covers_pipeline && Smem::covers_reduce &&
        Smem::covers_global_reduce && Smem::covers_write_result,
        "Marlin shared memory allocation is smaller than kernel scratch usage");
    static_assert(Smem::reduce_before_s_out, "Marlin reduction scratch overlaps pending s_out copy");

    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }
    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;
    int slice_count = 0;
    int slice_idx;

    if (slice_col_par >= n_tiles) {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
        C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
        locks += (slice_col_par / n_tiles) * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    auto init_slice = [&]() {
        // determine how many iterations
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
        if (slice_iters == 0) return;
        if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
        // number of slices and current slice_idx
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0) slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0)) {
                slice_idx = slice_count - 1;
            }
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0) {
                    slice_idx--;   
                }
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 8;
            C += 16 * thread_m_blocks * prob_n / 8;
            locks += n_tiles; slice_col = 0;
        }
    };
    init_slice();

    int a_gl_stride = prob_k / 8;
    constexpr int a_sh_stride = Smem::a_sh_stride;
    constexpr int a_gl_rd_delta_o = Smem::a_sh_stride;
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = Smem::a_sh_stage;
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);
    
    int b_gl_stride = prob_n / 8;
    constexpr int b_sh_stride = Smem::b_sh_stride;
    constexpr int true_b_sh_stride = Smem::true_b_sh_stride;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / true_b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int true_b_sh_stage = Smem::true_b_sh_stage;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;
    constexpr int true_b_sh_wr_iters = ceildiv(true_b_sh_stage, b_sh_wr_delta);

    constexpr int s_in_gl_stride = Smem::s_in_sh_stride;
    constexpr int s_in_sh_stride = Smem::s_in_sh_stride;
    constexpr int s_in_sh_rd_delta = 4 * (threads / b_sh_stride);
    constexpr int s_in_sh_stage = Smem::s_in_sh_stage;
    constexpr int s_in_gl_rd_delta = s_in_gl_stride;
    constexpr int s_out_sh_stride = Smem::s_out_sh_stride;

    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));
    int b_gl_rd = b_gl_stride * (threadIdx.x / true_b_sh_stride) + (threadIdx.x % true_b_sh_stride);
    b_gl_rd += true_b_sh_stride * slice_col; b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;
    int s_in_gl_rd = s_in_gl_rd_delta * slice_row + threadIdx.x;
    int s_in_sh_wr = threadIdx.x;
    int s_in_sh_rd = 4 * (threadIdx.x / b_sh_stride) + threadIdx.x % 4;
    int s_out_gl_rd = s_out_sh_stride * slice_col + threadIdx.x;
    int s_out_sh_wr = threadIdx.x;
    int s_out_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

    bool a_sh_wr_pred[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++) {
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
    }
    bool s_in_sh_wr_pred = use_s_in && threadIdx.x < s_in_sh_stride;
    bool s_out_sh_wr_pred = use_s_out && threadIdx.x < s_out_sh_stride;

    auto transform_a = [&](int i) { int row = i / a_gl_rd_delta_o; return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row; };
    int a_sh_wr_trans[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++) {
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    }
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
        #pragma unroll
        for (int j = 0; j < thread_m_blocks; j++) {
            a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
        }
    }
    // Since B-accesses have non-constant stride they have to be computed at runtime; we break dependicies between
    // subsequent accesses with a tile by maintining multiple pointers (we have enough registers), a tiny optimization.
    const int4* B_ptr[true_b_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < true_b_sh_wr_iters; i++) {
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;
    }

    extern __shared__ int4 sh[];
    // Shared memory storage for global fetch pipelines.
    int4* sh_a = sh + Smem::a_offset_int4;
    int4* sh_b = sh + Smem::b_offset_int4;
    int4* sh_s_in = sh + Smem::s_in_offset_int4;
    int4* sh_s_out = sh + Smem::s_out_offset_int4;

    alignas(int4) Vec<TVec2, 4> frag_a[2][thread_m_blocks];
    I1 frag_b_quant[2];
    alignas(int4) FragC frag_c[thread_m_blocks][4][2];
    alignas(int2) Vec<TVec2, 2> frag_s_in[2];
    alignas(int4) Vec<TVec2, 1> frag_s_out[8];

    // zero out all accum registers to 0
    auto zero_accums = [&]() {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++) {
            reinterpret_cast<float*>(frag_c)[i] = 0;
        }
    };

    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            // move activations A from GMEM to SMEM
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]],
                    &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                    a_sh_wr_pred[i]
                );
            }
            // move quatnized weights from GMEM to SMEM
            int4* sh_b_stage = sh_b + true_b_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < true_b_sh_wr_iters; i++) {
                if constexpr (true_b_sh_stage % b_sh_wr_delta == 0) {
                    cp_async4(
                        &sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                }
                else if (b_sh_wr_delta * i + b_sh_wr < true_b_sh_stage) {
                    cp_async4(
                        &sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                }
                B_ptr[i] += b_gl_rd_delta_o;
            }
            // copy input scales
            if (use_s_in) {
                int4* sh_s_stage = sh_s_in + s_in_sh_stage * pipe;
                if (s_in_sh_wr_pred) {
                    cp_async4_stream(
                        &sh_s_stage[s_in_sh_wr],
                        &s_in[s_in_gl_rd]
                    );
                }
                s_in_gl_rd += s_in_gl_rd_delta;
            }
        }
        cp_async_fence();
    };

    // ensure that 
    auto wait_for_stage = [&]() {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    auto fetch_to_registers = [&](int k, int pipe) {
        if (use_s_in) {
            int2* sh_s_stage = reinterpret_cast<int2*>(sh_s_in + s_in_sh_stage * pipe);
            reinterpret_cast<int2*>(&frag_s_in[k % 2])[0] = sh_s_stage[s_in_sh_rd_delta * (k % b_sh_wr_iters) + s_in_sh_rd];
        }
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++) {
            ldsm4(
                &frag_a[k % 2][i],
                &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]
            );
        }
        int4* sh_b_stage = sh_b + true_b_sh_stage * pipe;
        frag_b_quant[k % 2] = reinterpret_cast<I1*>(sh_b_stage)[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd];
    };

    auto matmul = [&](int k) {
        int b_quant = frag_b_quant[k % 2][0];
        Vec<TVec2, 2> scale;
        if constexpr (use_s_in) {
            scale = frag_s_in[k % 2];
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Vec<TVec2, 2> frag_b0;
            Vec<TVec2, 2> frag_b1;
            if constexpr (use_s_in) {
                frag_b0 = dequant_and_scale<T>(b_quant, scale, j * 2 + 0);
                frag_b1 = dequant_and_scale<T>(b_quant, scale, j * 2 + 1);
            }
            else {
                frag_b0 = dequant_unit<T>(b_quant, j * 2 + 0);
                frag_b1 = dequant_unit<T>(b_quant, j * 2 + 1);
            }
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                Traits::mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                Traits::mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
            #pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
                #pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
                        #pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                                #pragma unroll
                                for (int k = 0; k < 4; k++) {
                                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                                }
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
                    #pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
                        }
                    }
                }
                __syncthreads();
            }
        }
    };

    auto global_reduce = [&](bool first = false, bool last = false) {
        constexpr int active_threads = 32 * thread_n_blocks / 4;
        if (threadIdx.x < active_threads) {
            int c_gl_stride = prob_n / 8;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 4 * (active_threads / 32);
            int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
            c_gl_wr += (2 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads;
            int c_sh_wr = threadIdx.x; int row = (threadIdx.x % 32) / 4;
            if (!first) {
                #pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++) {
                    cp_async4_pred(
                        &sh[c_sh_wr + c_sh_wr_delta * i],
                        &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
                        i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
                    );
                }
                cp_async_fence();
                cp_async_wait<0>();
            }
            #pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++) {
                if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
                    if (!first) {
                        int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            float val = Traits::to_float(reinterpret_cast<T*>(&c_red)[j]);
                            reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += val;
                        }
                    }
                    if (!last) {
                        int4 c;
                        #pragma unroll
                        for (int j = 0; j < 2 * 4; j++) {
                            reinterpret_cast<T*>(&c)[j] = Traits::from_float(
                                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
                            );
                        }
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
                    }
                }
            }
        }
    };

    auto write_result = [&]() {
        int c_gl_stride = prob_n / 8;
        constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
        int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
        constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));
        int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
        int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
        c_sh_wr += 32 * (threadIdx.x / 32);
        int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
        int c_gl_wr_end = c_gl_stride * prob_m;

        auto write = [&](int idx, float c0, float c1, Vec<typename Traits::Vec2, 1>& s) {
            typename Traits::Vec2 res = Traits::make_vec2(c0, c1);
            if (use_s_out) res = Traits::mul2(res, s[0]);
            
            float res_low = Traits::get_low_float(res);
            float res_high = Traits::get_high_float(res);
            
            ((typename Traits::Vec2*)sh)[idx] = Traits::from_scalars(
                Traits::from_float(res_low),
                Traits::from_float(res_high)
            );
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4) {
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = c_sh_wr + 8 * j;
                    write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s_out[j * 2 + 0]);
                    write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s_out[j * 2 + 0]);
                    write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s_out[j * 2 + 1]);
                    write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s_out[j * 2 + 1]);
                }
                c_sh_wr += 16 * (4 * c_sh_stride);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
            if (c_gl_wr < c_gl_wr_end) {
                C[c_gl_wr] = sh[c_sh_rd];
                c_gl_wr += c_gl_wr_delta;
                c_sh_rd += c_sh_rd_delta;
            }
        }
    };

    auto start_pipes = [&]() {
        #pragma unroll
        for (int i = 0; i < stages - 1; i++) {
            fetch_to_shared(i, i, i < slice_iters);
        }
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    while (slice_iters) {
        #pragma unroll
        for (int pipe = 0; pipe < stages;) {
            #pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0) break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;
        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;
            if (use_s_out && last) {
                if (s_out_sh_wr_pred) {
                    cp_async4_stream(&sh_s_out[s_out_sh_wr], &s_out[s_out_gl_rd]);
                    cp_async_fence();
                }
            }
            thread_block_reduce();
            if (use_s_out && last) {
                cp_async_wait<0>();
                __syncthreads();
                if (threadIdx.x / 32 < thread_n_blocks / 4) {
                    reinterpret_cast<int4*>(&frag_s_out)[0] = sh_s_out[s_out_sh_rd + 0];
                    reinterpret_cast<int4*>(&frag_s_out)[1] = sh_s_out[s_out_sh_rd + 4];
                }
            }
            if (slice_count > 1) {
                barrier_acquire(&locks[slice_col], slice_idx);
                global_reduce(slice_idx == 0, last);
                barrier_release(&locks[slice_col], last);
            }
            if (last) {
                write_result();
            }
            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
                #pragma unroll
                for (int i = 0; i < true_b_sh_wr_iters; i++) {
                    B_ptr[i] += true_b_sh_stride - b_gl_rd_delta_o * k_tiles;
                }
                if (slice_col == 0) {
                    #pragma unroll
                    for (int i = 0; i < true_b_sh_wr_iters; i++) {
                        B_ptr[i] -= b_gl_stride;
                    }
                }
                s_in_gl_rd = threadIdx.x;
                s_out_gl_rd = s_out_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;
const int ERR_MAX_PAR = 3;
const int THREADS = 256;
const int STAGES = 4;
const int MARLIN_MIN_MAX_PAR = 32;

// --- Kernel Wrappers ---
template <
    typename T,
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const int stages,
    const bool use_s_in = true,
    const bool use_s_out = true
>
__global__ void Marlin_kernel(const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C, const int4* __restrict__ s_in, const int4* __restrict__ s_out, int prob_m, int prob_n, int prob_k, int* locks) {
    Marlin_impl<T, threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, use_s_in, use_s_out>(A, B, C, s_in, s_out, prob_m, prob_n, prob_k, locks);
}

template <
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const int stages,
    const bool use_s_in,
    const bool use_s_out
>
constexpr int marlin_shared_mem_bytes() {
    return MarlinSharedMemLayout<
        THREADS, thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, use_s_in, use_s_out
    >::bytes;
}

#define CALL_IF(T, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, USE_S_IN, USE_S_OUT) \
    else if (                                                                      \
        thread_m_blocks == THREAD_M_BLOCKS &&                                      \
        thread_n_blocks == THREAD_N_BLOCKS &&                                      \
        thread_k_blocks == THREAD_K_BLOCKS) {                                      \
        constexpr int SHARED_MEM_BYTES = marlin_shared_mem_bytes<THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, USE_S_IN, USE_S_OUT>(); \
        static bool attr_set = []() {                                              \
            cudaFuncSetAttribute(                                                  \
                Marlin_kernel<T, THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, USE_S_IN, USE_S_OUT>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
                SHARED_MEM_BYTES);                                                 \
            return true;                                                           \
        }();                                                                       \
        (void)attr_set;                                                            \
        Marlin_kernel<T, THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, USE_S_IN, USE_S_OUT> \
            <<<blocks, THREADS, SHARED_MEM_BYTES, stream>>>(                       \
                A_ptr, B_ptr, C_ptr, s_in_ptr, s_out_ptr,                          \
                prob_m, prob_n, prob_k,                                            \
                locks);                                                            \
    }

template <typename T>
int marlin_cuda_template(
    const void* A, const void* B, void* C, void* s_in, void* s_out,
    int prob_m, int prob_n, int prob_k, void* workspace,
    int dev, cudaStream_t stream, int thread_k, int thread_n, int sms, int max_par
) {
    int tot_m = prob_m;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad = 16 * tot_m_blocks - tot_m;
    if (max_par < MARLIN_MIN_MAX_PAR) return ERR_MAX_PAR;

    if (sms == -1) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }

    if (thread_k == -1 || thread_n == -1) {
        if (prob_m <= 16) {
            thread_k = 256;
            thread_n = 128;
        }
        else if (s_in == nullptr && prob_n >= 8192) {
            thread_k = 128;
            thread_n = 512;
        }
        else {
            thread_k = 128;
            thread_n = 256;
        }
    }

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
    int blocks = sms;

    if (prob_n % thread_n != 0 || prob_k % thread_k != 0) return ERR_PROB_SHAPE;
    if (prob_m == 0 || prob_n == 0 || prob_k == 0) return 0;

    const int4* A_ptr = (const int4*)A;
    const int4* B_ptr = (const int4*)B;
    int4* C_ptr = (int4*)C;
    const int4* s_in_ptr = (const int4*)s_in;
    const int4* s_out_ptr = (const int4*)s_out;
    int* locks = (int*)workspace;
    int ret = 0;

    for (int i = 0; i < tot_m_blocks; i += 4) {
        int thread_m_blocks = tot_m_blocks - i;
        prob_m = tot_m - 16 * i;
        int par = 1;
        if (thread_m_blocks > 4) {
            // Note that parallel > 1 currently only works for inputs without any padding
            par = (16 * thread_m_blocks - pad) / 64;
            if (par > max_par) {
                par = max_par;
            }
            prob_m = 64 * par;
            i += 4 * (par - 1);
            thread_m_blocks = 4;
        }

        // For compilation speed, we only define the kernel configurations that have seemed useful (in terms of performance)
        // in our testing, however many more are, in principle, possible.
        if (s_in_ptr != nullptr && s_out_ptr != nullptr) {
            if (false) {}
            CALL_IF(T, 1, 8, 16, true, true)
            CALL_IF(T, 1, 16, 8, true, true)
            CALL_IF(T, 2, 16, 8, true, true)
            CALL_IF(T, 3, 16, 8, true, true)
            CALL_IF(T, 4, 16, 8, true, true)
            else ret = ERR_KERN_SHAPE;
        } else if (s_in_ptr != nullptr) {
            if (false) {}
            CALL_IF(T, 1, 8, 16, true, false)
            CALL_IF(T, 1, 16, 8, true, false)
            CALL_IF(T, 2, 16, 8, true, false)
            CALL_IF(T, 3, 16, 8, true, false)
            CALL_IF(T, 4, 16, 8, true, false)
            else ret = ERR_KERN_SHAPE;
        } else if (s_out_ptr != nullptr) {
            if (false) {}
            CALL_IF(T, 1, 8, 16, false, true)
            CALL_IF(T, 1, 16, 8, false, true)
            CALL_IF(T, 2, 16, 8, false, true)
            CALL_IF(T, 3, 16, 8, false, true)
            CALL_IF(T, 4, 16, 8, false, true)
            CALL_IF(T, 1, 32, 8, false, true)
            CALL_IF(T, 2, 32, 8, false, true)
            CALL_IF(T, 3, 32, 8, false, true)
            CALL_IF(T, 4, 32, 8, false, true)
            else ret = ERR_KERN_SHAPE;
        } else {
            if (false) {}
            CALL_IF(T, 1, 8, 16, false, false)
            CALL_IF(T, 1, 16, 8, false, false)
            CALL_IF(T, 2, 16, 8, false, false)
            CALL_IF(T, 3, 16, 8, false, false)
            CALL_IF(T, 4, 16, 8, false, false)
            else ret = ERR_KERN_SHAPE;
        }

        A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
        C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
    }
    return ret;
}

template int marlin_cuda_template<half>(const void*, const void*, void*, void*, void*, int, int, int, void*, int, cudaStream_t, int, int, int, int);
template int marlin_cuda_template<__nv_bfloat16>(const void*, const void*, void*, void*, void*, int, int, int, void*, int, cudaStream_t, int, int, int, int);
