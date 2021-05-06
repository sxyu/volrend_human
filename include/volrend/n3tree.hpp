#pragma once

#include "volrend/common.hpp"
#include "volrend/data_format.hpp"

#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <utility>
#include "cnpy.h"

#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"

#ifdef VOLREND_CUDA
#include <cuda_fp16.h>
#else
#include <half.hpp>
using half_float::half;
#endif

namespace volrend {

struct _BBoxItem {
    _BBoxItem(float x, float y, float z, float size)
        : xyz{x, y, z}, size(size) {}
    // Center of voxel box
    float xyz[3];
    // 1/2 side length
    float size;
};

struct _LBSWeightItem {
    uint16_t index;
    __half weight;
};

struct _WarpGridItem {
    __half transform[12];
    float max_sigma;
};

// Read-only N3Tree loader
struct N3Tree {
    N3Tree();
    explicit N3Tree(const std::string& path, const std::string& rig_path = "");
    ~N3Tree();

    // Open npz
    void open(const std::string& path, const std::string& rig_path = "");
    // Open memory data stream (for web mostly)
    void open_mem(const char* data, uint64_t size,
                  const char* rig_data = nullptr, uint64_t rig_size = 0);

    // KinTree propagation
    void update_kintree();

    // Generate wireframe (returns line vertex positions; 9 * (a-b c-d) ..)
    // assignable to Mesh.vert
    // up to given depth (default none)
    std::vector<float> gen_wireframe(int max_depth = 100000) const;

    // Spatial branching factor
    int N = 0;
    // Size of data stored on each leaf
    int data_dim;
    // Data format (SH, SG etc)
    DataFormat data_format;
    // Capacity
    int capacity = 0;

    // ** LBS/Skeleton
    // Number of joints
    int n_joints;
    // Number of vertices
    int n_verts;
    // **

    // Translation
    std::array<float, 3> offset;
    // Translation for current posed space
    mutable std::array<float, 3> offset_pose;
    // Scaling for coordinates
    std::array<float, 3> scale;
    // Scaling for current posed space
    mutable std::array<float, 3> scale_pose;

    // Axis-angle pose at each joint, set by user (only used if rigged)
    std::vector<glm::vec3> pose;
    std::vector<glm::mat4> pose_mats;
    // Root translation
    glm::vec3 trans;

    // True if rigged
    bool is_rigged();

    bool is_data_loaded();
#ifdef VOLREND_CUDA
    bool is_cuda_loaded();
#endif

    // Clear the CPU memory.
    void clear_cpu_memory();

    // Index pack/unpack
    int pack_index(int nd, int i, int j, int k);
    std::tuple<int, int, int, int> unpack_index(int packed);

    // NDC config
    bool use_ndc;
    float ndc_width, ndc_height, ndc_focal;
    glm::vec3 ndc_avg_up, ndc_avg_back, ndc_avg_cen;

    // ***** Internal ******
#ifdef VOLREND_CUDA
    // CUDA memory
    mutable struct {
        __half* data = nullptr;
        int32_t* child = nullptr;
        float* offset = nullptr;
        float* offset_pose = nullptr;
        float* scale = nullptr;
        float* scale_pose = nullptr;
        float* extra = nullptr;

        // BBOX and scale at each leaf
        _BBoxItem* bbox = nullptr;
        uint64_t* inv_ptr = nullptr;

        // Per-leaf LBS weight from NN
        uint32_t* lbs_weight_start = nullptr;
        _LBSWeightItem* lbs_weight = nullptr;
        // Computed warp grid
        _WarpGridItem* warp = nullptr;
        uint8_t* warp_dist_map = nullptr;
        // Joint transform
        float* joint_transform = nullptr;
    } device;
#endif

    // Main data holder
    mutable cnpy::NpyArray data_;

    // Child link data holder
    cnpy::NpyArray child_;

    // Optional extra data
    cnpy::NpyArray extra_;

    // ** LBS / Skeleton
    // LBS weight data holder (SMPL weights)
    cnpy::NpyArray weights_;

    // Joints positions at rest (SMPL J)
    std::vector<glm::vec3> joint_pos_;

    // Vertex positions (SMPL v_template)
    cnpy::NpyArray v_template_;

    // Kinectic tree parents (SMPL kintree_table; only nonempty if rigged)
    std::vector<uint32_t> kintree_table_;

    // **

    // Bounding box center and size for leaf voxel at given compressed leaf
    // index (Lx4 : cen_x cen_y cen_z scale)
    std::vector<_BBoxItem> bbox_;
    // Pointer to data (/data_dim) for given compressed leaf index (L)
    std::vector<uint64_t> inv_ptr_;

    std::vector<uint32_t> lbs_weight_start_;
    std::vector<_LBSWeightItem> lbs_weight_;
    std::vector<glm::vec3> joint_pos_posed_;
    mutable std::vector<float> joint_transform_;

    int N2_, N3_;

   private:
    void load_npz(cnpy::npz_t& npz);
    void load_rig_npz(cnpy::npz_t& npz);
    void gen_bbox();

    // Paths
    std::string npz_path_, rig_path_, poses_bounds_path_;
    bool data_loaded_;

    mutable float last_sigma_thresh_;

#ifdef VOLREND_CUDA
    bool cuda_loaded_;
    void load_cuda();
    void free_cuda();
    void update_kintree_cuda();
#endif
};

}  // namespace volrend
