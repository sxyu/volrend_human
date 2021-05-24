#include "volrend/n3tree.hpp"
#include "volrend/data_format.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <atomic>
#include <thread>
#include "cnpy.h"

#include "glm/geometric.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/mat3x3.hpp"
#include <nanoflann.hpp>

#ifndef VOLREND_CUDA
#include "half.hpp"
#endif

namespace volrend {
namespace {
struct KdTreePointsAdaptor {
    KdTreePointsAdaptor(const float* pts_ptr, unsigned int n_pts)
        : pts_(pts_ptr), n_pts_(n_pts) {}
    inline unsigned int kdtree_get_point_count() const { return n_pts_; }

    inline float kdtree_distance(const float* p1, const unsigned int index_2,
                                 unsigned int size) const {
        const float* p2 = &pts_[index_2 * 3];
        return sqrtf((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                     (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                     (p1[2] - p2[2]) * (p1[2] - p2[2]));
    }
    inline float kdtree_get_pt(const unsigned int i, int dim) const {
        return pts_[i * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

   private:
    const float* pts_;
    const unsigned int n_pts_;
};

// Compute LBS weights from mesh surface using nearest-neighbors
void _nn_weights_cpu(const std::vector<_BBoxItem>& bbox,
                     const cnpy::NpyArray& v_template,
                     const cnpy::NpyArray& weights,
                     std::vector<uint32_t>& lbs_weight_start_out,
                     std::vector<_LBSWeightItem>& lbs_weight_out) {
    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, KdTreePointsAdaptor>,
        KdTreePointsAdaptor, 3>;

    KdTreePointsAdaptor adaptor(v_template.data<float>(), v_template.shape[0]);
    KdTree kdtree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    const size_t n_points = bbox.size();
    const float* weights_ptr = weights.data<float>();
    const int n_joints = weights.shape[1];

    std::vector<uint32_t> nn_id(n_points);

    {
        std::atomic<uint32_t> point_id(0);
        const unsigned int n_thd = std::thread::hardware_concurrency();
        auto worker = [&]() {
            size_t index;
            double dist;
            nanoflann::KNNResultSet<double> resultSet(1);
            while (true) {
                uint32_t i = point_id++;
                if (i >= n_points) break;
                resultSet.init(&index, &dist);

                kdtree.findNeighbors(resultSet, bbox[i].xyz,
                                     nanoflann::SearchParams(10));
                const float* in_weights = weights_ptr + index * n_joints;
                nn_id[i] = index;
            }
        };
        std::vector<std::thread> thds;
        for (unsigned int i = 0; i < n_thd; ++i) thds.emplace_back(worker);
        for (unsigned int i = 0; i < n_thd; ++i) thds[i].join();
        lbs_weight_out.reserve(n_points * 0.04f);
        lbs_weight_start_out.resize(n_points + 1);
        for (uint32_t i = 0; i < n_points; ++i) {
            const float* in_weights = weights_ptr + nn_id[i] * n_joints;
            lbs_weight_start_out[i] = (uint32_t)lbs_weight_out.size();
            for (int j = 0; j < n_joints; ++j) {
                if (in_weights[j] > 1e-6f) {
                    lbs_weight_out.push_back(
                        _LBSWeightItem{(uint16_t)j, in_weights[j]});
                }
            }
        }
        lbs_weight_start_out.back() = lbs_weight_out.size();
    }
}

// Load LBS weights from --weights file
void _copy_sparse_weights_cpu(const std::vector<uint64_t>& inv_ptr,
                              const cnpy::NpyArray& valid_joint_ids,
                              const cnpy::NpyArray& weights,
                              std::vector<uint32_t>& lbs_weight_start_out,
                              std::vector<_LBSWeightItem>& lbs_weight_out) {
    const size_t n_points = inv_ptr.size();
    const int n_valid_joints = valid_joint_ids.shape[0];
    const __half* weights_ptr = weights.data<__half>();
    const int64_t* valid_joint_map = valid_joint_ids.data<int64_t>();
    lbs_weight_out.reserve(n_points * 0.04f);
    lbs_weight_start_out.resize(n_points + 1);
    for (uint32_t i = 0; i < n_points; ++i) {
        const __half* in_weights = weights_ptr + inv_ptr[i] * n_valid_joints;
        lbs_weight_start_out[i] = (uint32_t)lbs_weight_out.size();
        for (int j = 0; j < n_valid_joints; ++j) {
            if (in_weights[j] > 1e-6f) {
                lbs_weight_out.push_back(_LBSWeightItem{
                    (uint16_t)valid_joint_map[j], in_weights[j]});
            }
        }
    }
    lbs_weight_start_out.back() = lbs_weight_out.size();
}
}  // namespace

void N3Tree::update_kintree() {
    if (pose.empty()) return;
    std::vector<glm::mat4> pose_mats(pose.size(), glm::mat4(1.0f));
    joint_transform_.resize(12 * pose.size());
    for (size_t i = 0; i < pose.size(); ++i) {
        const float angle = glm::length(pose[i]);
        if (std::fabs(angle) < 1e-6) {
            pose_mats[i] = glm::mat4(1);
        } else {
            pose_mats[i] = glm::rotate(pose_mats[i], angle, pose[i] / angle);
        }
    }

    if (pose_canon_l.size()) {
        for (size_t i = 0; i < pose.size(); ++i) {
            pose_mats[i] = glm::mat4(pose_canon_l[i]) * pose_mats[i] *
                           glm::mat4(pose_canon_r[i]);
        }
    }

    pose_mats[0][3] = glm::vec4(joint_pos_[0] + trans, 1.f);
    for (size_t i = 1; i < pose.size(); ++i) {
        pose_mats[i][3] =
            glm::vec4(joint_pos_[i] - joint_pos_[kintree_table_[i]], 1.f);
        if (~kintree_table_[i]) {
            pose_mats[i] = pose_mats[kintree_table_[i]] * pose_mats[i];
        }
    }

    joint_pos_posed_.resize(n_joints);
    this->pose_mats = pose_mats;
    for (size_t i = 0; i < pose.size(); ++i) {
        joint_pos_posed_[i] = pose_mats[i][3];
        pose_mats[i][3] -=
            glm::vec4(glm::mat3(pose_mats[i]) * joint_pos_[i], 0.f);
    }
    for (size_t i = 0; i < pose.size(); ++i) {
        float* jtrans_ptr = joint_transform_.data() + 12 * i;
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                jtrans_ptr[j * 3 + k] = pose_mats[i][j][k];
            }
        }
    }

#ifdef VOLREND_CUDA
    update_kintree_cuda();
#endif
}

void N3Tree::load_rig_npz(cnpy::npz_t& npz, cnpy::npz_t& weights_npz) {
    bool use_given_lbs_weights =
        weights_npz.count("weights") && weights_npz.count("valid_joint_ids");
    {
        const auto& kt_node = npz["kintree_table"];
        n_joints = kt_node.shape[1];
        kintree_table_.resize(n_joints);
        if (kt_node.word_size == 8) {
            std::copy(kt_node.data<int64_t>(),
                      kt_node.data<int64_t>() + n_joints,
                      kintree_table_.begin());
        } else if (kt_node.word_size == 4) {
            std::copy(kt_node.data<uint32_t>(),
                      kt_node.data<uint32_t>() + n_joints,
                      kintree_table_.begin());
        }
    }
    {
        auto& jnt_node = npz["J"];
        if (jnt_node.shape[0] != n_joints || jnt_node.shape[1] != 3 ||
            jnt_node.word_size != 4) {
            std::cerr << "RIG LOAD ERROR: Invalid model J\n";
        }
        const float* jnt_ptr = jnt_node.data<float>();
        joint_pos_.resize(n_joints);
        joint_pos_on_load_.resize(n_joints);
        for (int i = 0; i < n_joints; ++i) {
            const float* jnti_ptr = jnt_ptr + i * 3;
            joint_pos_[i] = glm::vec3(jnti_ptr[0], jnti_ptr[1], jnti_ptr[2]);
            joint_pos_on_load_[i] = joint_pos_[i];
        }
    }
    pose.resize(n_joints);
    pose_canon.resize(n_joints);
    trans = glm::vec3(0);
    std::fill(pose.begin(), pose.end(), glm::vec3(0));
    std::fill(pose_canon.begin(), pose_canon.end(), glm::vec3(0));

    gen_bbox();
    if (use_given_lbs_weights) {
        auto& valid_joint_ids = weights_npz["valid_joint_ids"];
        const int n_valid_joints = valid_joint_ids.shape[0];
        if (valid_joint_ids.shape.size() != 1 ||
            valid_joint_ids.word_size != 8) {
            std::cerr << "rig weight load error: invalid valid_joint_ids\n";
            std::exit(1);
        }

        auto& weights = weights_npz["weights"];
        if (weights.shape.size() != 5 || weights.shape[0] != capacity ||
            weights.shape[1] != N || weights.shape[2] != N ||
            weights.shape[3] != N || weights.shape[4] != n_valid_joints ||
            weights.word_size != 2) {
            std::cerr << "rig weight load error: invalid lbs weights\n";
            std::exit(1);
        }

        _copy_sparse_weights_cpu(inv_ptr_, valid_joint_ids, weights,
                                 lbs_weight_start_, lbs_weight_);
    } else {
        std::cerr
            << "WARNING: using automatic nearest-neighbor weights from "
               "mesh surface, did you specify the correct --weights file?\n";
        auto& v_template = npz["v_template"];
        int n_verts = v_template.shape[0];
        if (v_template.shape[1] != 3 || v_template.word_size != 4) {
            std::cerr << "rig load error: invalid model v_template\n";
            std::exit(1);
        }
        auto& weights = npz["weights"];
        if (weights.shape[0] != n_verts || weights.shape[1] != n_joints ||
            weights.word_size != 4) {
            std::cerr << "rig load error: invalid model weights\n";
            std::exit(1);
        }
        _nn_weights_cpu(bbox_, v_template, weights, lbs_weight_start_,
                        lbs_weight_);
    }
}

void N3Tree::load_canon(const std::string& path) {
    if (!std::ifstream(path)) {
        std::cerr << "canon load error: file could not be opened\n";
        std::exit(1);
    }
    cnpy::npz_t npz = cnpy::npz_load(path);
    if (npz.count("left") == 0 || npz.count("right") == 0) {
        std::cerr << "canon load error: left/right array not present\n";
        std::exit(1);
    }

    auto& left = npz["left"];
    auto& right = npz["right"];
    if (left.shape.size() != 3 || right.shape.size() != 3 ||
        left.shape[0] != n_joints || right.shape[0] != n_joints ||
        left.shape[1] != 3 || right.shape[1] != 3 || left.shape[2] != 3 ||
        right.shape[2] != 3 || left.word_size != 4 || right.word_size != 4) {
        std::cerr << "canon load error: invalid left or right array, "
                     "must be type float32, size [n_joints, 3, 3].\n";
        std::exit(1);
    }

    pose_canon_l.resize(n_joints);
    pose_canon_r.resize(n_joints);
    const float* left_ptr = left.data<float>();
    const float* right_ptr = right.data<float>();
    for (int i = 0; i < n_joints; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                pose_canon_l[i][k][j] = left_ptr[j * 3 + k];
                pose_canon_r[i][k][j] = right_ptr[j * 3 + k];
            }
        }
        left_ptr += 9;
        right_ptr += 9;

        uint32_t p = kintree_table_[i];
        if (~p) {
            joint_pos_[i] = pose_canon_l[i] * (joint_pos_on_load_[i] -
                                               joint_pos_on_load_[p]) +
                            joint_pos_[p];
        }
        glm::quat rot_q =
            glm::quat_cast(glm::transpose(pose_canon_l[i] * pose_canon_r[i]));
        pose_canon[i] = glm::axis(rot_q) * glm::angle(rot_q);
    }
}

void N3Tree::load_pose(const std::string& path) {
    if (!std::ifstream(path)) {
        std::cerr << "pose load error: file could not be opened\n";
        std::exit(1);
    }
    std::ifstream pose_ifs(path);
    for (int i = 0; i < pose.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            pose_ifs >> pose[i][j];
        }
    }
}

namespace {
void _gen_bbox_impl(const N3Tree& tree, size_t nodeid, size_t xi, size_t yi,
                    size_t zi, size_t gridsz, std::vector<_BBoxItem>& out,
                    std::vector<uint64_t>& inv_ptr) {
    const int32_t* child = tree.child_.data<int32_t>() + nodeid * tree.N3_;
    const __half* data =
        tree.data_.data<__half>() + nodeid * tree.N3_ * tree.data_dim;
    int cnt = 0;
    // Use integer coords to avoid precision issues
    for (size_t i = xi * tree.N; i < (xi + 1) * tree.N; ++i) {
        for (size_t j = yi * tree.N; j < (yi + 1) * tree.N; ++j) {
            for (size_t k = zi * tree.N; k < (zi + 1) * tree.N; ++k) {
                if (child[cnt] == 0) {
                    // Add this cube
                    if (data[tree.data_dim - 1] > 1e-2) {
                        out.push_back(
                            {((float)(i + 0.5) / gridsz - tree.offset[0]) /
                                 tree.scale[0],
                             ((float)(j + 0.5) / gridsz - tree.offset[1]) /
                                 tree.scale[1],
                             ((float)(k + 0.5) / gridsz - tree.offset[2]) /
                                 tree.scale[2],
                             0.5f / gridsz});
                        // 1.f / gridsz / tree.scale[0],
                        // 1.f / gridsz / tree.scale[1],
                        // 1.f / gridsz / tree.scale[2]
                        inv_ptr.push_back(nodeid * tree.N3_ + cnt);
                    }
                } else {
                    _gen_bbox_impl(tree, nodeid + child[cnt], i, j, k,
                                   gridsz * tree.N, out, inv_ptr);
                }
                ++cnt;
            }
        }
    }
}
}  // namespace

void N3Tree::gen_bbox() {
    const size_t size_estimate = int((capacity * N3_ * 7) / 8) + 1;
    bbox_.reserve(size_estimate * 4);
    inv_ptr_.reserve(size_estimate);
    _gen_bbox_impl(*this, 0, 0, 0, 0, N, bbox_, inv_ptr_);
}

bool N3Tree::is_rigged() { return kintree_table_.size() > 0; }

}  // namespace volrend
