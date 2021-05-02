#pragma once
#include "volrend/common.hpp"
#include <cstdio>
#include <cmath>

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
    scalar_t norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
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

__device__ __inline__ unsigned int expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// 3D Morton code
__device__ __inline__ unsigned int morton_code_3(unsigned int x, unsigned y, unsigned z) {
    unsigned int xx = expand_bits(x);
    unsigned int yy = expand_bits(y);
    unsigned int zz = expand_bits(z);
    return xx * 4 + yy * 2 + zz;
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
