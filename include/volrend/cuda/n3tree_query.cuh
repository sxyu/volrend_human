#pragma once

#define N3_WARP_GRID_DEPTH 8
#define N3_WARP_GRID_SIZE (1 << N3_WARP_GRID_DEPTH)

#include "volrend/camera.hpp"
#include "volrend/cuda/common.cuh"
#include <cuda_fp16.h>
#include "volrend/cuda/data_spec.cuh"

namespace volrend {

__host__ __device__ __inline__ static void query_single_from_root(
                                      const TreeSpec& tree,
                                      float* VOLREND_RESTRICT xyz,
                                      const half** VOLREND_RESTRICT out,
                                      float* VOLREND_RESTRICT cube_sz) {
    const float fN = tree.N;
    xyz[0] = max(min(xyz[0], 1.f - 1e-6f), 0.f);
    xyz[1] = max(min(xyz[1], 1.f - 1e-6f), 0.f);
    xyz[2] = max(min(xyz[2], 1.f - 1e-6f), 0.f);
    int64_t ptr = 0;
    *cube_sz = fN;
    while (true) {
        // Find index of query point, in {0, ... N^3}
        // float index = 4.f * (xyz[2] > 0.5f) + 2.f * (xyz[1] > 0.5f) + (xyz[0] > 0.5f);
        float index = 0.f;
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            xyz[i] *= fN;
            const float idx_dimi = floorf(xyz[i]);
            index = index * fN + idx_dimi;
            xyz[i] -= idx_dimi;
        }

        // Find child offset
        const int64_t sub_ptr = ptr + (int32_t)index;
        const int64_t skip = tree.child[sub_ptr];

        // Add to output
        if (skip == 0/* || *cube_sz >= max_cube_sz*/) {
            *out = tree.data + sub_ptr * tree.data_dim;
            break;
        }
        *cube_sz *= fN;

        ptr += skip * tree.N * tree.N * tree.N;
    }
}

}  // namespace volrend
