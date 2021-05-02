#pragma once
#include "volrend/common.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>

#ifdef VOLREND_CUDA

#include <cuda_runtime.h>

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define N_BLOCKS_NEEDED(Q, N_CUDA_THREADS) ((Q - 1) / N_CUDA_THREADS + 1)

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm(
        const scalar_t* __restrict__ dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm2(
        const scalar_t* __restrict__ dir) {
    return dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _dist2(
        const scalar_t* __restrict__ a, const scalar_t* __restrict__ b) {
    return (a[0] - b[0]) * (a[0] - b[0]) +(a[1] - b[1]) * (a[1] - b[1]) +
           (a[2] - b[2]) * (a[2] - b[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _normalize(
        scalar_t* __restrict__ dir) {
    scalar_t invnorm = 1.f / _norm(dir);
    dir[0] *= invnorm; dir[1] *= invnorm; dir[2] *= invnorm;
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _mv3(
        const scalar_t* __restrict__ m,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out) {
    out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
    out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
    out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _mv_affine(
        const scalar_t* __restrict__ m,
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out) {
    out[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2] + m[9];
    out[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2] + m[10];
    out[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2] + m[11];
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _copy3(
        const scalar_t* __restrict__ v,
        scalar_t* __restrict__ out) {
    out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _dot3(
        const scalar_t* __restrict__ u,
        const scalar_t* __restrict__ v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

template<typename scalar_t>
__host__ __device__ __inline__ static
void _cross3(const scalar_t* a, const scalar_t* b, scalar_t* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template<typename scalar_t>
__host__ __device__ __inline__ static
float _detr_3x3(const scalar_t* m) {
    float a = m[4] * m[8] - m[7] * m[5];
    float b = m[7] * m[2] - m[1] * m[8];
    float c = m[1] * m[5] - m[4] * m[2];
    return m[0] * a + m[3] * b + m[6] * c;
}

template<typename scalar_t>
__host__ __device__ __inline__ static
bool _inv_affine_cm12(const scalar_t* m, scalar_t* out) {
    out[0] = m[4] * m[8] - m[7] * m[5];
    out[1] = m[7] * m[2] - m[1] * m[8];
    out[2] = m[1] * m[5] - m[4] * m[2];
    const float detr = m[0] * out[0] + m[3] * out[1] + m[6] * out[2];
    if (fabsf(detr) == 0.0f) return false;
    const float idetr = scalar_t(1) / detr;
    out[0] *= idetr; out[1] *= idetr; out[2] *= idetr;
    out[3] = (m[6] * m[5] - m[3] * m[8]) * idetr;
    out[4] = (m[0] * m[8] - m[6] * m[2]) * idetr;
    out[5] = (m[3] * m[2] - m[0] * m[5]) * idetr;
    out[6] = (m[3] * m[7] - m[6] * m[4]) * idetr;
    out[7] = (m[6] * m[1] - m[0] * m[7]) * idetr;
    out[8] = (m[0] * m[4] - m[3] * m[1]) * idetr;
    out[9] = -out[0] * m[9] - out[3] * m[10] - out[6] * m[11];
    out[10] = -out[1] * m[9] - out[4] * m[10] - out[7] * m[11];
    out[11] = -out[2] * m[9] - out[5] * m[10] - out[8] * m[11];
    return true;
}

template <typename scalar_t>
__device__ __inline__ bool outside_grid(const scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        if (q[i] < 0.0 || q[i] >= 1.0 - 1e-10)
            return true;
    }
    return false;
}

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           const scalar_t* __restrict__ scale) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + scale[i] * q[i];
    }
}

__host__ __device__ __inline__ uint32_t _expand_bits(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__host__ __device__ __inline__ uint32_t _unexpand_bits(uint32_t v) {
    v &= 0x49249249;
    v = (v | (v >> 2)) & 0xc30c30c3;
    v = (v | (v >> 4)) & 0xf00f00f;
    v = (v | (v >> 8)) & 0xff0000ff;
    v = (v | (v >> 16)) & 0x0000ffff;
    return v;
}

// 3D Morton code
__host__ __device__ __inline__ uint32_t morton_code_3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = _expand_bits(x);
    uint32_t yy = _expand_bits(y);
    uint32_t zz = _expand_bits(z);
    return (xx << 2) + (yy << 1) + zz;
}

__host__ __device__ __inline__ void inv_morton_code_3(uint32_t code,
        uint32_t* x, uint32_t* y, uint32_t* z) {
    *x = _unexpand_bits(code >> 2);
    *y = _unexpand_bits(code >> 1);
    *z = _unexpand_bits(code);
}

namespace volrend {

// Beware that NVCC doesn't work with C files and __VA_ARGS__
cudaError_t cuda_assert(const cudaError_t code, const char* const file,
                        const int line, const bool abort);

}  // namespace volrend

#define cuda(...) cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, true);

#else
#define cuda(...)
#endif
