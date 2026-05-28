// Copyright (c) 2026 Samsung Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/macros/Macros.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// --- CUDA Checks ---
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F16(x) TORCH_CHECK(x.scalar_type() == torch::kFloat16, #x " must be float16 tensor")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == torch::kBFloat16, #x " must be bfloat16 tensor")
#define CHECK_F32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " tensor dtype must be float32")
#define CHECK_INT8(x) TORCH_CHECK(x.scalar_type() == torch::kInt8, #x " tensor dtype must be int8")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == torch::kInt32, #x " tensor dtype must be int32")

#define CHECK_CUDA_CONT_F16(x) \
    do {                       \
        CHECK_CUDA(x);         \
        CHECK_CONTIGUOUS(x);   \
        CHECK_F16(x);          \
    } while (0)
#define CHECK_CUDA_CONT_BF16(x) \
    do {                        \
        CHECK_CUDA(x);          \
        CHECK_CONTIGUOUS(x);    \
        CHECK_BF16(x);          \
    } while (0)
#define CHECK_CUDA_CONT_F32(x) \
    do {                       \
        CHECK_CUDA(x);         \
        CHECK_CONTIGUOUS(x);   \
        CHECK_F32(x);          \
    } while (0)
#define CHECK_CUDA_CONT_INT8(x) \
    do {                        \
        CHECK_CUDA(x);          \
        CHECK_CONTIGUOUS(x);    \
        CHECK_INT8(x);          \
    } while (0)
#define CHECK_CUDA_CONT_INT32(x) \
    do {                         \
        CHECK_CUDA(x);           \
        CHECK_CONTIGUOUS(x);     \
        CHECK_INT32(x);          \
    } while (0)

// --- Constants ---
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;  // Default for dynamic kernels
constexpr int WARP_SIZE = 32;
#define HALF_FLT_MAX 65504.F
#define FINAL_MASK 0xffffffff

// --- Half Precision Utilities ---
static __device__ __forceinline__ uint32_t half2_to_uint32(const half2 &h) {
    return reinterpret_cast<const uint32_t &>(h);
}
static __device__ __forceinline__ half2 uint32_to_half2(uint32_t v) {
    return reinterpret_cast<const half2 &>(v);
}

// --- BFloat16 Precision Utilities ---
static __device__ __forceinline__ uint32_t bf162_to_uint32(const __nv_bfloat162 &h) {
    return reinterpret_cast<const uint32_t &>(h);
}
static __device__ __forceinline__ __nv_bfloat162 uint32_to_bf162(uint32_t v) {
    return reinterpret_cast<const __nv_bfloat162 &>(v);
}

#define SIGN_MASK_LO(b) ((b) ? 0u : 0x00008000u)
#define SIGN_MASK_HI(b) ((b) ? 0u : 0x80000000u)

// --- General Utilities ---
#define DIV_UP(a, b) (((a) + (b)-1) / (b))
template <typename T>
__device__ __forceinline__ T clamp_inf_for_half(const float input) {
    return static_cast<T>(input);
}
template <>
__device__ __forceinline__ half clamp_inf_for_half(const float input) {
    return __float2half(fminf(fmaxf(input, -64504.0f), 64504.0f));
}
template <>
__device__ __forceinline__ __nv_bfloat16 clamp_inf_for_half(const float input) {
    return __float2bfloat16(fminf(fmaxf(input, -3.3895313892515355e+38f), 3.3895313892515355e+38f));
}

// Sign-flipping utility (for GEMV kernels)
constexpr uint32_t SIGN_MASK = 0x80008000u;
__device__ __forceinline__ uint4 apply_sign(uint4 x4, uint32_t bits, int shift_base) {
    // compute the four masks directly, but reuse the shifted bits value
    uint32_t shifted = bits << shift_base;
    x4.x ^= shifted & SIGN_MASK;
    x4.y ^= (shifted << 1) & SIGN_MASK;
    x4.z ^= (shifted << 2) & SIGN_MASK;
    x4.w ^= (shifted << 3) & SIGN_MASK;
    return x4;
}

// --- Reduction Utilities (Common for Linear Kernels) ---

// Generic add operation
template <typename T>
inline __device__ T reduce_add(T a, T b) {
    return a + b;
}
template <>
inline __device__ half reduce_add(half a, half b) {
    return __hadd(a, b);
}
template <>
inline __device__ half2 reduce_add(half2 a, half2 b) {
    return __hadd2(a, b);
}
template <>
inline __device__ __nv_bfloat16 reduce_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}
template <>
inline __device__ __nv_bfloat162 reduce_add(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hadd2(a, b);
}

constexpr unsigned FULL_MASK = 0xffffffff;
__device__ __forceinline__ half2 shfl_down_half2(half2 v, int off, unsigned mask = FULL_MASK) {
    uint32_t x = *reinterpret_cast<uint32_t *>(&v);
    x = __shfl_down_sync(mask, x, off);
    half2 out = *reinterpret_cast<half2 *>(&x);
    return out;
}

__device__ __forceinline__ __nv_bfloat162 shfl_down_bf162(__nv_bfloat162 v, int off, unsigned mask = FULL_MASK) {
    uint32_t x = *reinterpret_cast<uint32_t *>(&v);
    x = __shfl_down_sync(mask, x, off);
    __nv_bfloat162 out = *reinterpret_cast<__nv_bfloat162 *>(&x);
    return out;
}

__device__ __forceinline__ half warp_reduce_sum_half2_ret_half(half2 r) {
    // canonical tree: 16,8,4,2,1
    for (int off = 16; off > 0; off >>= 1) {
        r = __hadd2(r, shfl_down_half2(r, off));
    }
    // lane 0 now holds sum of all lanes
    return __hadd(__low2half(r), __high2half(r));
}

__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf162_ret_bf16(__nv_bfloat162 r) {
    // canonical tree: 16,8,4,2,1
    for (int off = 16; off > 0; off >>= 1) {
        r = __hadd2(r, shfl_down_bf162(r, off));
    }
    // lane 0 now holds sum of all lanes
    return __hadd(__low2bfloat16(r), __high2bfloat16(r));
}

// Warp-level reduction
template <typename T, unsigned int Size = WARP_SIZE>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = Size / 2; mask > 0; mask >>= 1) {
        val = reduce_add(val, __shfl_xor_sync(FINAL_MASK, val, mask));
    }
    return val;
}

// Block-level reduction for float
template <unsigned int block_size>
__inline__ __device__ float blockReduceSum(float val) {
    if (block_size == WARP_SIZE) {
        return warpReduceSum<float>(val);
    }

    __shared__ float shared[block_size / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum<float>(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < block_size / WARP_SIZE) ? shared[threadIdx.x] : 0.0f;
        val = warpReduceSum<float, block_size / WARP_SIZE>(val);
    }
    return val;  // Result in lane 0 of warp 0
}

// Block-level reduction for half2
template <unsigned int block_size>
__inline__ __device__ float blockReduceSum(half2 val_fma) {
    float val = __half2float(val_fma.x) + __half2float(val_fma.y);

    return blockReduceSum<block_size>(val);
}

// Block-level reduction for bfloat162
template <unsigned int block_size>
__inline__ __device__ float blockReduceSum(__nv_bfloat162 val_fma) {
    float val = __bfloat162float(val_fma.x) + __bfloat162float(val_fma.y);

    return blockReduceSum<block_size>(val);
}
