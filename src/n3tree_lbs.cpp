#include "volrend/n3tree.hpp"
#include "volrend/data_format.hpp"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <atomic>
#include <thread>

#include "glm/geometric.hpp"
#include "glm/gtc/matrix_transform.hpp"
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

void N3Tree::load_rig_npz(cnpy::npz_t& npz) {
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
        for (int i = 0; i < n_joints; ++i) {
            const float* jnti_ptr = jnt_ptr + i * 3;
            joint_pos_[i] = glm::vec3(jnti_ptr[0], jnti_ptr[1], jnti_ptr[2]);
        }
    }
    {
        auto& vt_node = npz["v_template"];
        n_verts = vt_node.shape[0];
        if (vt_node.shape[1] != 3 || vt_node.word_size != 4) {
            std::cerr << "RIG LOAD ERROR: Invalid model v_template\n";
        }
        std::swap(v_template_, vt_node);
    }
    {
        auto& wt_node = npz["weights"];
        if (wt_node.shape[0] != n_verts || wt_node.shape[1] != n_joints ||
            wt_node.word_size != 4) {
            std::cerr << "RIG LOAD ERROR: Invalid model weights\n";
        }
        std::swap(weights_, wt_node);
    }
    pose.resize(n_joints);
    trans = glm::vec3(0);
    std::fill(pose.begin(), pose.end(), glm::vec3(0));

    gen_bbox();
    _nn_weights_cpu(bbox_, v_template_, weights_, lbs_weight_start_,
                    lbs_weight_);
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
