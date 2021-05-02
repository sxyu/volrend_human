#include "volrend/n3tree.hpp"

#include <limits>
#include <cstdio>
#include <cassert>
#include <thread>
#include <atomic>
#include <chrono>

#include "volrend/cuda/n3tree_query.cuh"
#include <cuda_fp16.h>

#define BEGIN_PROFILE  auto start = std::chrono::high_resolution_clock::now()
#define PROFILE( \
    x)  do{printf("%s: %f ms\n", #x, std::chrono::duration<double, \
        std::milli>(std::chrono::high_resolution_clock::now() - \
        start).count()); start = std::chrono::high_resolution_clock::now(); \
        }while(false)

namespace volrend {
namespace {
const int N_CUDA_THREADS = 512;

namespace device {
__global__ static void lbs_transform_kernel(TreeSpec tree) {
    CUDA_GET_THREAD_ID(idx, tree.n_leaves_compr);
    float inv_lbs_trans[12];
    int coords[3];
    {
        float lbs_trans[12];
        float targ_pos[3];

        for (int i = 0; i < 12; ++i) lbs_trans[i] = 0.f;
        for (int i = tree.lbs_weight_start[idx]; i < tree.lbs_weight_start[idx + 1]; ++i) {
            const _LBSWeightItem& it = tree.lbs_weight[i];
            const float wt = __half2float(it.weight);
            const float* trans_in = tree.joint_transform + 12 * it.index;
            for (int i = 0; i < 12; ++i) lbs_trans[i] += trans_in[i] * wt;
        }

        _mv_affine(lbs_trans, tree.bbox[idx].xyz, targ_pos);
        transform_coord(targ_pos, tree.offset, tree.scale);

        coords[0] = floorf(targ_pos[0] * N3_WARP_GRID_SIZE);
        coords[1] = floorf(targ_pos[1] * N3_WARP_GRID_SIZE);
        coords[2] = floorf(targ_pos[2] * N3_WARP_GRID_SIZE);

        if (coords[0] < 0 || coords[1] < 0 || coords[2] < 0 ||
            coords[0] >= N3_WARP_GRID_SIZE || coords[1] >= N3_WARP_GRID_SIZE ||
            coords[2] >= N3_WARP_GRID_SIZE) {
            // Out of bounds
            return;
        }
        if (!_inv_affine_cm12(lbs_trans, inv_lbs_trans)) {
            // Singular
            return;
        }
    }
    const uint32_t warp_id = morton_code_3(coords[0], coords[1], coords[2]);
    _WarpGridItem& warp_out = tree.warp[warp_id];
    half sigma = tree.data[(tree.inv_ptr[idx] + 1) * tree.data_dim - 1];
    const float max_sigma = __half2float(warp_out.max_sigma);
    if (max_sigma < __half2float(sigma)) {
        warp_out.max_sigma = sigma;
        for (int i = 0; i < 12; ++i) {
            warp_out.transform[i] = __float2half(inv_lbs_trans[i]);
        }

        if (max_sigma == 0.0f) {
            uint32_t warp_subidx = warp_id;
            uint32_t subgrid_off = 0;
            uint32_t subgrid_sz = N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE;
            do {
                tree.warp_dist_map[subgrid_off + warp_subidx] = 0;
                warp_subidx >>= 3;
                subgrid_off += subgrid_sz;
                subgrid_sz >>= 3;
            } while (subgrid_sz > 0);
        }
    }
}  // void nn_weights_kernel

__global__ static void warp_dist_prop_kernel(TreeSpec tree) {
    CUDA_GET_THREAD_ID(warp_id, N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE);
    uint32_t cnt = 0;
    uint32_t warp_subidx = warp_id;
    uint32_t subgrid_off = 0;
    uint32_t subgrid_sz = N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE;
    do {
        if (tree.warp_dist_map[subgrid_off + warp_subidx] != 255) break;
        warp_subidx >>= 3;
        subgrid_off += subgrid_sz;
        subgrid_sz >>= 3;
        ++cnt;
    } while (subgrid_sz > 0);
    tree.warp_dist_map[warp_id] = (uint8_t) cnt;
}

__global__ static void debug_lbs_draw_kernel(TreeSpec tree) {
    CUDA_GET_THREAD_ID(idx, tree.n_leaves_compr);
    half* data_ptr = tree.data + tree.inv_ptr[idx] * tree.data_dim;

    // Visualize LBS transform max sigma
    // float xyz[3];
    // _copy3(tree.bbox[idx].xyz, xyz);
    // transform_coord(xyz, tree.offset, tree.scale);
    // int coords[3];
    // coords[0] = floorf(xyz[0] * N3_WARP_GRID_SIZE);
    // coords[1] = floorf(xyz[1] * N3_WARP_GRID_SIZE);
    // coords[2] = floorf(xyz[2] * N3_WARP_GRID_SIZE);
    // if (coords[0] < 0 || coords[1] < 0 || coords[2] < 0 ||
    //         coords[0] >= N3_WARP_GRID_SIZE || coords[1] >= N3_WARP_GRID_SIZE ||
    //         coords[2] >= N3_WARP_GRID_SIZE) {
    //     // Out of bounds
    //     return;
    // }
    // const uint64_t warp_idx = uint64_t(coords[0]) * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE +
    //     coords[1] * N3_WARP_GRID_SIZE + coords[2];
    // const _WarpGridItem& __restrict__ warp_out = tree.warp[warp_idx];
    // float msigma = __half2float(warp_out.max_sigma);
    // data_ptr[3] = warp_out.max_sigma;

    // Visualize LBS weights
    // int maxd = 0;
    // float maxdv = 0;
    // for (int i = tree.lbs_weight_start[idx]; i < tree.lbs_weight_start[idx + 1]; ++i) {
    //     if (__half2float(tree.lbs_weight[i].weight) > maxdv) {
    //         maxdv = __half2float(tree.lbs_weight[i].weight);
    //         maxd = tree.lbs_weight[i].index;
    //     }
    // }
    //
    // // DRAW LBS MAX FOR DEBUGGING
    // maxd = maxd & 3;
    // if (maxd == 0){
    //     data_ptr[0] = 0.f;
    //     data_ptr[1] = 1.f;
    //     data_ptr[2] = 0.f;
    // } else if (maxd == 1){
    //     data_ptr[0] = 1.f;
    //     data_ptr[1] = 0.f;
    //     data_ptr[2] = 0.f;
    // } else if (maxd == 2){
    //     data_ptr[0] = 0.f;
    //     data_ptr[1] = 0.f;
    //     data_ptr[2] = 1.f;
    // } else if (maxd == 3){
    //     data_ptr[0] = 1.f;
    //     data_ptr[1] = 0.f;
    //     data_ptr[2] = 1.f;
    // }
}  // void nn_weights_kernel
}  // namespace device
}  // namespace

void N3Tree::load_cuda() {
    free_cuda();

    const size_t data_sz = (size_t) capacity * N3_ * data_dim * sizeof(half);
    const size_t child_sz = (size_t) capacity * N3_ * sizeof(int32_t);
    cuda(Malloc((void**)&device.data, data_sz));
    cuda(Malloc((void**)&device.child, child_sz));
    if (device.offset == nullptr) {
        cuda(Malloc((void**)&device.offset, 3 * sizeof(float)));
    }
    if (device.scale == nullptr) {
        cuda(Malloc((void**)&device.scale, 3 * sizeof(float)));
    }
    cuda(MemcpyAsync(device.child, child_.data<int32_t>(), child_sz,
                cudaMemcpyHostToDevice));
    const half* data_ptr = this->data_.data<half>();
    cuda(MemcpyAsync(device.data, data_ptr, data_sz,
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.offset, offset.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(device.scale, scale.data(), 3 * sizeof(float),
                cudaMemcpyHostToDevice));
    if (bbox_.size()) {
        const size_t bb_size = bbox_.size() * sizeof(bbox_[0]);
        cuda(Malloc((void**)&device.bbox, bb_size));
        cuda(MemcpyAsync(device.bbox, bbox_.data(), bb_size,
                    cudaMemcpyHostToDevice));

        const size_t iptr_size = inv_ptr_.size() * sizeof(inv_ptr_[0]);
        cuda(Malloc((void**)&device.inv_ptr, iptr_size));
        cuda(MemcpyAsync(device.inv_ptr, inv_ptr_.data(), iptr_size,
                    cudaMemcpyHostToDevice));
    }
    if (extra_.data_holder.size()) {
        cuda(Malloc((void**)&device.extra, extra_.data_holder.size()));
        cuda(MemcpyAsync(device.extra, extra_.data<float>(),
                    extra_.data_holder.size(),
                    cudaMemcpyHostToDevice));
    } else {
        device.extra = nullptr;
    }

    cuda_loaded_ = true;
}

void N3Tree::update_kintree_cuda() {
    if (kintree_table_.empty()) return;

    // Load LBS weight
    if (device.lbs_weight_start == nullptr) {
        const size_t size = (inv_ptr_.size() + 1) * sizeof(lbs_weight_start_[0]);
        cuda(Malloc((void**)&device.lbs_weight_start, size));
        cuda(MemcpyAsync(device.lbs_weight_start, lbs_weight_start_.data(),
                    size,
                    cudaMemcpyHostToDevice));
    }
    if (device.lbs_weight == nullptr) {
        const size_t size = lbs_weight_.size() * sizeof(lbs_weight_[0]);
        cuda(Malloc((void**)&device.lbs_weight, size));
        cuda(MemcpyAsync(device.lbs_weight, lbs_weight_.data(),
                    size,
                    cudaMemcpyHostToDevice));
    }
    const size_t warp_size = sizeof(_WarpGridItem) * N3_WARP_GRID_SIZE *
        N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE;
    if (device.warp == nullptr) {
        cuda(Malloc((void**)&device.warp, warp_size));
    }
    const size_t warp_dist_map_size = 2 * N3_WARP_GRID_SIZE *
        N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE;
    if (device.warp_dist_map == nullptr) {
        cuda(Malloc((void**)&device.warp_dist_map, warp_dist_map_size));
    }
    {
        const size_t size = 12 * n_joints * sizeof(float);
        if (device.joint_transform == nullptr) {
            cuda(Malloc((void**)&device.joint_transform, size));
        }
        cuda(MemcpyAsync(device.joint_transform, joint_transform_.data(),
                    size,
                    cudaMemcpyHostToDevice));
    }

    // TODO: remove this memset by recording the nonzero entries
    cuda(Memset(device.warp, 0, warp_size));
    cuda(Memset(device.warp_dist_map, -1, warp_dist_map_size));
    {
        const int blocks = N_BLOCKS_NEEDED(inv_ptr_.size(), N_CUDA_THREADS);
        device::lbs_transform_kernel<<<blocks, N_CUDA_THREADS>>>(*this);
    }

    {
        const int blocks = N_BLOCKS_NEEDED(warp_dist_map_size / 2, N_CUDA_THREADS);
        device::warp_dist_prop_kernel<<<blocks, N_CUDA_THREADS>>>(*this);
    }
    // For debugging / verification
    {
        // const int blocks = N_BLOCKS_NEEDED(inv_ptr_.size(), N_CUDA_THREADS);
        // device::debug_lbs_draw_kernel<<<blocks, N_CUDA_THREADS>>>(*this);
    }
}

void N3Tree::free_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    if (device.offset != nullptr) cuda(Free(device.offset));
    if (device.scale != nullptr) cuda(Free(device.scale));
    if (device.extra != nullptr) cuda(Free(device.extra));

    if (device.lbs_weight_start != nullptr) cuda(Free(device.lbs_weight_start));
    if (device.lbs_weight != nullptr) cuda(Free(device.lbs_weight));
    if (device.warp != nullptr) cuda(Free(device.warp));
    if (device.warp_dist_map != nullptr) cuda(Free(device.warp_dist_map));
    if (device.joint_transform != nullptr) cuda(Free(device.joint_transform));
    if (device.bbox != nullptr) cuda(Free(device.bbox));
    if (device.inv_ptr != nullptr) cuda(Free(device.inv_ptr));
}
}  // namespace volrend
