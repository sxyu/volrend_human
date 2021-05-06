#include "volrend/n3tree.hpp"

#include <limits>
#include <cstdio>
#include <cassert>
#include <thread>
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
__global__ static void set_zero_kernel(TreeSpec tree) {
    CUDA_GET_THREAD_ID(warp_id, N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE);
    tree.warp[warp_id].max_sigma = 0.0;
}

__global__ void _lbs_scale_kernel(TreeSpec tree) {
    // This kernel stores min/max bounding box of transformed voxels
    // within tree.scale_pose / tree.offset_pose resp.

    // Go through canonical spane leaf voxels
    CUDA_GET_THREAD_ID(idx, tree.n_leaves_compr);
    {
        half sigma = tree.data[(tree.inv_ptr[idx] + 1) * tree.data_dim - 1];
        if (__half2float(sigma) < 1e-2) return;
    }
    // Compute LBS transform at this voxel
    float lbs_trans[12];
    float inv_lbs_trans[12];
    for (int i = 0; i < 12; ++i) lbs_trans[i] = 0.f;
    for (int i = tree.lbs_weight_start[idx]; i < tree.lbs_weight_start[idx + 1]; ++i) {
        const _LBSWeightItem& it = tree.lbs_weight[i];
        const float wt = __half2float(it.weight);
        const float* trans_in = tree.joint_transform + 12 * it.index;
        for (int i = 0; i < 12; ++i) lbs_trans[i] += trans_in[i] * wt;
    }
    if (!_inv_affine_cm12(lbs_trans, inv_lbs_trans)) {
        // Singular
        return;
    }

    // Transform it to the posed space
    int coords[3];
    float src_pos[3];
    float targ_pos[3];
    const _BBoxItem& __restrict__ bbox = tree.bbox[idx];
    _mv_affine(lbs_trans, bbox.xyz, targ_pos);

    // TODO reduce the number of atomics, use shared memory
    for (int i = 0; i < 3; ++i) {
        atomicMinf(&tree.scale_pose[i], targ_pos[i]);
        atomicMaxf(&tree.offset_pose[i], targ_pos[i]);
    }
}

__host__ void lbs_scale_launcher(const N3Tree& tree) {
    // Determine offset_pose and scale_pose
    float max_init3[3] = {-1e9, -1e9, -1e9};
    float min_init3[3] = {1e9, 1e9, 1e9};

    cuda(MemcpyAsync(tree.device.scale_pose, min_init3, sizeof(min_init3), cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(tree.device.offset_pose, max_init3, sizeof(max_init3), cudaMemcpyHostToDevice));

    const int leaf_blocks = N_BLOCKS_NEEDED(tree.inv_ptr_.size(), N_CUDA_THREADS);
    _lbs_scale_kernel<<<leaf_blocks, N_CUDA_THREADS>>>(tree);

    cuda(MemcpyAsync(min_init3, tree.device.scale_pose, sizeof(min_init3), cudaMemcpyDeviceToHost));
    cuda(MemcpyAsync(max_init3, tree.device.offset_pose, sizeof(max_init3), cudaMemcpyDeviceToHost));

    float center[3], radius[3];
    float max_radius = 0.f;
    for (int i = 0; i < 3; ++i) {
        center[i] = (max_init3[i] + min_init3[i]) * 0.5f;
        radius[i] = (max_init3[i] - min_init3[i]) * 0.5f;
        max_radius = std::max(radius[i], max_radius);
    }
    for (int i = 0; i < 3; ++i) {
        radius[i] = max_radius;
        tree.scale_pose[i] = 0.5f / radius[i];
        tree.offset_pose[i] = 0.5f * (1.0f - center[i] / radius[i]);
    }

    cuda(MemcpyAsync(tree.device.scale_pose, tree.scale_pose.data(),
                3 * sizeof(tree.scale_pose[0]), cudaMemcpyHostToDevice));
    cuda(MemcpyAsync(tree.device.offset_pose, tree.offset_pose.data(),
                3 * sizeof(tree.offset_pose[0]), cudaMemcpyHostToDevice));
}

__global__ void lbs_transform_kernel(TreeSpec tree) {
    // Go through canonical space leaf voxels
    CUDA_GET_THREAD_ID(idx, tree.n_leaves_compr);
    // Compute LBS transform at this voxel
    float lbs_trans[12];
    float inv_lbs_trans[12];
    for (int i = 0; i < 12; ++i) lbs_trans[i] = 0.f;
    for (int i = tree.lbs_weight_start[idx]; i < tree.lbs_weight_start[idx + 1]; ++i) {
        const _LBSWeightItem& it = tree.lbs_weight[i];
        const float wt = __half2float(it.weight);
        const float* trans_in = tree.joint_transform + 12 * it.index;
        for (int i = 0; i < 12; ++i) lbs_trans[i] += trans_in[i] * wt;
    }

    // Invert the LBS transform
    if (!_inv_affine_cm12(lbs_trans, inv_lbs_trans)) {
        // Singular
        return;
    }
    const float sigma = __half2float(tree.data[(tree.inv_ptr[idx] + 1) * tree.data_dim - 1]);

    // Transform some points within the voxel into pose space
    // and set the transform in tree.warp, the inverse warp grid
    // (We do it for several points per voxel to reduce the number
    //  of holes caused by splatting)
    int coords[3];
    float src_pos[3];
    float targ_pos[3];
    constexpr int HALF_N_POINTS = 1;
    const _BBoxItem& __restrict__ bbox = tree.bbox[idx];
    float dx = tree.bbox[idx].size * (1.f / (tree.scale[0] * (HALF_N_POINTS + 1)));
    float dy = tree.bbox[idx].size * (1.f / (tree.scale[1] * (HALF_N_POINTS + 1)));
    float dz = tree.bbox[idx].size * (1.f / (tree.scale[2] * (HALF_N_POINTS + 1)));
    for (int i = -HALF_N_POINTS; i <= HALF_N_POINTS; ++i) {
        src_pos[0] = i * dx + bbox.xyz[0];
        for (int j = -HALF_N_POINTS; j <= HALF_N_POINTS; ++j) {
            src_pos[1] = j * dy + bbox.xyz[1];
            for (int k = -HALF_N_POINTS; k <= HALF_N_POINTS; ++k) {
                src_pos[2] = k * dz + bbox.xyz[2];
                // Transform it to the posed space
                _mv_affine(lbs_trans, src_pos, targ_pos);

                transform_coord(targ_pos, tree.offset_pose, tree.scale_pose);
                coords[0] = floorf(targ_pos[0] * N3_WARP_GRID_SIZE);
                coords[1] = floorf(targ_pos[1] * N3_WARP_GRID_SIZE);
                coords[2] = floorf(targ_pos[2] * N3_WARP_GRID_SIZE);

                // TODO: bounds right now are same as rest pose bounds.
                // If you move the human too much parts of the body will disappear,
                // due to going out of bounds.
                // Ideally the bounds should be auto-updated with each kintree update.
                if (coords[0] < 0 || coords[1] < 0 || coords[2] < 0 ||
                        coords[0] >= N3_WARP_GRID_SIZE || coords[1] >= N3_WARP_GRID_SIZE ||
                        coords[2] >= N3_WARP_GRID_SIZE) {
                    // Out of bounds
                    continue;
                }

                const uint32_t warp_id = morton_code_3(coords[0], coords[1], coords[2]);
                _WarpGridItem& warp_out = tree.warp[warp_id];
                // Only keep the voxel with highest density
                float old_max_sigma = atomicMaxf(&warp_out.max_sigma, sigma);
                if (old_max_sigma < sigma) {
                    // Set the warp in the posed space warp grid
                    for (int i = 0; i < 12; ++i) {
                        warp_out.transform[i] = __float2half(inv_lbs_trans[i]);
                    }

                    if (tree.warp_dist_map[warp_id] == 255) {
                        // In warp_dist_map, store 0 if each voxel is occupied.
                        // Also propagates to parent voxels in the implicit tree.
                        // (Note warp_dist_map was set to 255 uniformly before this kernel)
                        // Also see the comment above the kernel below.
                        uint32_t warp_subidx = warp_id;
                        uint32_t subgrid_off = 0;
                        uint32_t subgrid_sz = N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE;
                        do {
                            tree.warp_dist_map[subgrid_off + warp_subidx] = 0;
                            warp_subidx >>= 3;
                            subgrid_off += subgrid_sz;
                            subgrid_sz >>= 3;
                        } while (subgrid_sz > 0);
                    }  // if warp_dist_map == 255
                }  // if max_sigma < sigma
            }  // for k
        }  // for j
    }  // for i


}  // void lbs_transform_kernel

// This kernel propagates the warp_dist_map, which stores
// the smallest i s.t. the size 2^i 'octree voxel' is nonempty around the point.
// used for accelerating rendering.

// warp_dist_map is stored as an linear tree structure of
// hierarchical sections as follows (concatenated):
// [256^3, 128^3, 64^3, ..., 1^3]
// Each level is stored using Morton code.
// For element n, the parent is n / 8 in the next section.

// Initially (after lbs_transform_kernel), each tree node is 0 if any
// child voxel is occupied, or 255 else.
// The function below makes the first 256^3 section store the distance to the
// first ancestor which has an occupied child, for use when rendering.
// The rest of the tree is small and just left unused after this.
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

// This unused kernel is for debugging purposes
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
    // const uint32_t warp_id = morton_code_3(coords[0], coords[1], coords[2]);
    // const _WarpGridItem& __restrict__ warp_out = tree.warp[warp_id];
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
}
}  // namespace device
}  // namespace

void N3Tree::load_cuda() {
    free_cuda();

    const size_t data_sz = (size_t) capacity * N3_ * data_dim * sizeof(half);
    const size_t child_sz = (size_t) capacity * N3_ * sizeof(int32_t);
    cuda(Malloc((void**)&device.data, data_sz));
    cuda(Malloc((void**)&device.child, child_sz));
    cuda(Malloc((void**)&device.offset, 3 * sizeof(float)));
    cuda(Malloc((void**)&device.scale, 3 * sizeof(float)));
    cuda(Malloc((void**)&device.offset_pose, 3 * sizeof(float)));
    cuda(Malloc((void**)&device.scale_pose, 3 * sizeof(float)));
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
    const size_t warp_dist_map_size = 1.2 * N3_WARP_GRID_SIZE *
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

    cuda(Memset(device.warp, 0, warp_size));
    const int leaf_blocks = N_BLOCKS_NEEDED(inv_ptr_.size(), N_CUDA_THREADS);
    const int warp_3d_blocks = N_BLOCKS_NEEDED(N3_WARP_GRID_SIZE * N3_WARP_GRID_SIZE *
            N3_WARP_GRID_SIZE, N_CUDA_THREADS);
    device::set_zero_kernel<<<warp_3d_blocks, N_CUDA_THREADS>>>(*this);
    device::lbs_scale_launcher(*this);
    cuda(Memset(device.warp_dist_map, -1, warp_dist_map_size));
    device::lbs_transform_kernel<<<leaf_blocks, N_CUDA_THREADS>>>(*this);
    device::warp_dist_prop_kernel<<<warp_3d_blocks, N_CUDA_THREADS>>>(*this);
    // For debugging / verification
    {
        // const int blocks = N_BLOCKS_NEEDED(inv_ptr_.size(), N_CUDA_THREADS);
        // device::debug_lbs_draw_kernel<<<leaf_blocks, N_CUDA_THREADS>>>(*this);
    }
}

void N3Tree::free_cuda() {
    if (device.data != nullptr) cuda(Free(device.data));
    if (device.child != nullptr) cuda(Free(device.child));
    if (device.offset != nullptr) cuda(Free(device.offset));
    if (device.offset_pose != nullptr) cuda(Free(device.offset_pose));
    if (device.scale != nullptr) cuda(Free(device.scale));
    if (device.scale_pose != nullptr) cuda(Free(device.scale_pose));
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
