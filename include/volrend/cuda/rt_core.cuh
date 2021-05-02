#pragma once
#include "volrend/cuda/n3tree_query.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/render_options.hpp"
#include "volrend/cuda/common.cuh"
#include "volrend/cuda/data_spec.cuh"
#include "volrend/cuda/lumisphere.cuh"

namespace volrend {
namespace device {
namespace {

template<typename scalar_t>
__device__ __inline__ void _dda_world(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax,
        const float* __restrict__ render_bbox) {
    scalar_t t1, t2;
    *tmin = 0.0;
    *tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = (render_bbox[i] + 1e-6 - cen[i]) * _invdir[i];
        t2 = (render_bbox[i + 3] - 1e-6 - cen[i]) * _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

template<typename scalar_t>
__device__ __inline__ scalar_t _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir) {
    scalar_t t1, t2;
    scalar_t tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template<typename scalar_t>
__device__ __inline__ void trace_ray(
        const TreeSpec& __restrict__ tree,
        scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        const scalar_t* __restrict__ cen,
        RenderOptions opt,
        float tmax_bg,
        scalar_t* __restrict__ out) {

    const float delta_scale = _get_delta_scale(
            tree.scale, /*modifies*/ dir);
    tmax_bg /= delta_scale;

    scalar_t tmin, tmax;
    scalar_t invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    float cen_tr[3], dir_tr[3];
    for (int i = 0; i < 3; ++i) {
        cen_tr[i] = tree.offset[i] + tree.scale[i] * cen[i];
        dir_tr[i] = dir[i] / tree.scale[i];
    }
    _dda_world(cen_tr, invdir, &tmin, &tmax, opt.render_bbox);
    tmax = min(tmax, tmax_bg);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        if (opt.render_depth)
            out[3] = 1.f;
        return;
    } else {
        int coords[3];
        scalar_t pos[3], pos_orig[3], tmp;
        const half* tree_val;
        // scalar_t basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        // maybe_precalc_basis(tree, vdir, basis_fn);
        // for (int i = 0; i < opt.basis_minmax[0]; ++i) {
        //     basis_fn[i] = 0.f;
        // }
        // for (int i = opt.basis_minmax[1] + 1; i < VOLREND_GLOBAL_BASIS_MAX; ++i) {
        //     basis_fn[i] = 0.f;
        // }

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        const scalar_t delta_t = opt.step_size * delta_scale;
        while (t < tmax) {
            // for (int j = 0; j < 3; ++j) {
            //     pos_orig[j] = pos[j] = cen[j] + t * dir_tr[j];
            //     pos[j] = cen_tr[j] + t * dir[j];
            //     coords[j] = floorf(pos[j] * N3_WARP_GRID_SIZE);
            // }
            pos[0] = cen_tr[0] + t * dir[0];
            pos[1] = cen_tr[1] + t * dir[1];
            pos[2] = cen_tr[2] + t * dir[2];

            scalar_t sigma = 1e9;
            if (tree.n_joints > 0) {
                for (int j = 0; j < 3; ++j)  {
                    pos_orig[j] = cen[j] + t * dir_tr[j];
                    pos[j] *= N3_WARP_GRID_SIZE;
                    coords[j] = floorf(pos[j]);
                }
                const uint64_t warp_idx = uint64_t(coords[0]) * N3_WARP_GRID_SIZE *
                    N3_WARP_GRID_SIZE +
                    coords[1] * N3_WARP_GRID_SIZE + coords[2];
                const _WarpGridItem& __restrict__ warp = tree.warp[warp_idx];
                if (__half2float(warp.max_sigma) < opt.sigma_thresh) {
                    t += opt.step_size;
                    continue;
                }

                // TODO: trilinear
                const half* m = warp.transform;
                for (int j = 0; j < 3; ++j) {
                    pos[j] = __half2float(m[j]) * pos_orig[0] +
                        __half2float(m[3 + j]) * pos_orig[1] +
                        __half2float(m[6 + j]) * pos_orig[2] +
                        __half2float(m[9 + j]);
                    pos[j] = tree.offset[j] + tree.scale[j] * pos[j];
                }
                sigma = __half2float(warp.max_sigma);
            }

            query_single_from_root(tree, pos, &tree_val, &cube_sz);

            // const scalar_t t_subcube = _dda_unit(pos, invdir) / cube_sz;
            sigma = min(sigma, __half2float(tree_val[tree.data_dim - 1]));
            if (sigma > opt.sigma_thresh) {
                scalar_t att = expf(-delta_t * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                if (opt.render_depth) {
                    out[0] += weight * t;
                } else {
//                     if (tree.data_format.basis_dim >= 0) {
//                         int off = 0;
// #define MUL_BASIS_I(v) basis_fn[v] * __half2float(tree_val[off + v])
// #pragma unroll 3
//                         for (int u = 0; u < 3; ++ u) {
//                             tmp = basis_fn[0] * __half2float(tree_val[off]);
//                             switch(tree.data_format.basis_dim) {
//                                 case 25:
//                                     tmp += MUL_BASIS_I(16) +
//                                         MUL_BASIS_I(17) +
//                                         MUL_BASIS_I(18) +
//                                         MUL_BASIS_I(19) +
//                                         MUL_BASIS_I(20) +
//                                         MUL_BASIS_I(21) +
//                                         MUL_BASIS_I(22) +
//                                         MUL_BASIS_I(23) +
//                                         MUL_BASIS_I(24);
//                                 case 16:
//                                     tmp += MUL_BASIS_I(9) +
//                                           MUL_BASIS_I(10) +
//                                           MUL_BASIS_I(11) +
//                                           MUL_BASIS_I(12) +
//                                           MUL_BASIS_I(13) +
//                                           MUL_BASIS_I(14) +
//                                           MUL_BASIS_I(15);
//
//                                 case 9:
//                                     tmp += MUL_BASIS_I(4) +
//                                         MUL_BASIS_I(5) +
//                                         MUL_BASIS_I(6) +
//                                         MUL_BASIS_I(7) +
//                                         MUL_BASIS_I(8);
//
//                                 case 4:
//                                     tmp += MUL_BASIS_I(1) +
//                                         MUL_BASIS_I(2) +
//                                         MUL_BASIS_I(3);
//                             }
//                             out[u] += weight / (1.f + expf(-tmp));
//                             off += tree.data_format.basis_dim;
//                         }
// #undef MUL_BASIS_I
//                     } else {
                        out[0] += __half2float(tree_val[0]) * weight;
                        out[1] += __half2float(tree_val[1]) * weight;
                        out[2] += __half2float(tree_val[2]) * weight;
                    // }
                }

                light_intensity *= att;

                if (light_intensity < opt.stop_thresh) {
                    // Almost full opacity, stop
                    if (opt.render_depth) {
                        out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
                    }
                    scalar_t scale = 1.f / (1.f - light_intensity);
                    out[0] *= scale; out[1] *= scale; out[2] *= scale;
                    out[3] = 1.f;
                    return;
                }
            }
            t += opt.step_size;
        }
        if (opt.render_depth) {
            out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
            out[3] = 1.f;
        } else {
            out[3] = 1.f - light_intensity;
        }
    }
}

}  // namespace
}  // namespace device
}  // namespace volrend
