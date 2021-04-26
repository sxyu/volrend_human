#pragma once

#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"

#ifdef __CUDACC__
#define VOLREND_RESTRICT __restrict__
#else
#define VOLREND_RESTRICT
#endif

namespace volrend {
namespace {

struct CameraSpec {
    const int width;
    const int height;
    const float fx, fy;
    const float* VOLREND_RESTRICT transform;
    CameraSpec(const Camera& camera) : width(camera.width),
        height(camera.height), fx(camera.fx), fy(camera.fy),
        transform(camera.device.transform) {
    }
};
struct TreeSpec {
    __half* VOLREND_RESTRICT const data;
    const int32_t* VOLREND_RESTRICT const child;
    const float* VOLREND_RESTRICT const offset;
    const float* VOLREND_RESTRICT const scale;
    const float* VOLREND_RESTRICT const extra;
    const _BBoxItem* VOLREND_RESTRICT const bbox;
    const uint64_t* VOLREND_RESTRICT const inv_ptr;
    const uint32_t* VOLREND_RESTRICT const lbs_weight_start;
    const _LBSWeightItem* VOLREND_RESTRICT const lbs_weight;
    _WarpGridItem* VOLREND_RESTRICT const warp;
    const float* VOLREND_RESTRICT const joint_transform;
    const int N;
    const int capacity;
    const int data_dim;
    const DataFormat data_format;
    const float ndc_width;
    const float ndc_height;
    const float ndc_focal;
    const int n_leaves_compr;
    const int n_verts;
    const int n_joints;

    TreeSpec(const N3Tree& tree, bool cpu=false) :
        data(cpu ? tree.data_.data<__half>() : tree.device.data),
        child(cpu ? tree.child_.data<int32_t>() : tree.device.child),
        offset(cpu ? tree.offset.data() : tree.device.offset),
        scale(cpu ? tree.scale.data() : tree.device.scale),
        extra(cpu ? tree.extra_.data<float>() : tree.device.extra),
        bbox(cpu ? tree.bbox_.data() : tree.device.bbox),
        inv_ptr(cpu ? tree.inv_ptr_.data() : tree.device.inv_ptr),
        lbs_weight_start(cpu ? tree.lbs_weight_start_.data() : tree.device.lbs_weight_start),
        lbs_weight(cpu ? tree.lbs_weight_.data() : tree.device.lbs_weight),
        warp(cpu ? nullptr : tree.device.warp),
        joint_transform(cpu ? tree.joint_transform_.data() : tree.device.joint_transform),
        N(tree.N),
        capacity(tree.capacity),
        data_dim(tree.data_dim),
        data_format(tree.data_format),
        ndc_width(tree.use_ndc ? tree.ndc_width : -1),
        ndc_height(tree.ndc_height),
        ndc_focal(tree.ndc_focal),
        n_leaves_compr(tree.inv_ptr_.size()),
        n_verts(tree.n_verts),
        n_joints(tree.n_joints) { }
};

}  // namespace
}  // namespace volrend
