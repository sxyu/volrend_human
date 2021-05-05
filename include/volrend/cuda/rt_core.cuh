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
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    float cen_local[3], dir_world[3];
    for (int i = 0; i < 3; ++i) {
        cen_local[i] = tree.offset[i] + tree.scale[i] * cen[i];
        dir_world[i] = dir[i] / tree.scale[i];
    }
    _dda_world(cen_local, invdir, &tmin, &tmax, opt.render_bbox);
    tmax = min(tmax, tmax_bg);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        if (opt.render_depth)
            out[3] = 1.f;
        return;
    } else {
        uint32_t coords[3];
        uint32_t ul[3];
        scalar_t pos[3], pos_world[3];//, tmp;
        const half* tree_val;

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        const scalar_t delta_t = opt.step_size * delta_scale;
        while (t < tmax) {
            pos[0] = cen_local[0] + t * dir[0];
            pos[1] = cen_local[1] + t * dir[1];
            pos[2] = cen_local[2] + t * dir[2];

            scalar_t sigma = 1e9;
            if (tree.n_joints > 0) {
#pragma unroll 3
                for (int j = 0; j < 3; ++j)  {
                    pos_world[j] = cen[j] + t * dir_world[j];
                    pos[j] *= N3_WARP_GRID_SIZE;
                    coords[j] = floorf(pos[j]);
                }
                const uint32_t warp_id = morton_code_3(coords[0], coords[1], coords[2]);

                const uint8_t warp_dist = tree.warp_dist_map[warp_id];
                if (warp_dist > 0) {
                    // This warp cell in pose space is empty, so we
                    // skip for an appropriate distance
                    // according to the computed warp_dist_map
                    const uint32_t cell_scale = (1 << (uint32_t)(warp_dist - 1));
                    const float inv_cell_scale = 1.f / cell_scale;
                    const uint32_t cell_ul_shift = 3 * (warp_dist - 1);

                    inv_morton_code_3((warp_id >> cell_ul_shift) << cell_ul_shift,
                        ul, ul + 1, ul + 2);
#pragma unroll 3
                    for (int j = 0; j < 3; ++j)  {
                        pos[j] = (pos[j] - ul[j]) * inv_cell_scale;
                    }

                    const scalar_t t_subcube = _dda_unit(pos, invdir);
                    const scalar_t delta_t = t_subcube * (float(cell_scale) / N3_WARP_GRID_SIZE);
                    t += delta_t + opt.step_size;
                    continue;
                }
                const _WarpGridItem& __restrict__ warp = tree.warp[warp_id];
                const uint32_t c000    = morton_code_3(coords[0]+1, coords[1], coords[2]);
                _WarpGridItem& warp_out000 = tree.warp[c000];

                if (coords[0] < 1 || coords[1] < 1 || coords[2] < 1 ||
                    coords[0] >= N3_WARP_GRID_SIZE - 1 || coords[1] >= N3_WARP_GRID_SIZE - 1 ||
                    coords[2] >= N3_WARP_GRID_SIZE - 1) {   // Out of bounds, don't interpolate
                    const half* m = warp.transform;
                    for (int j = 0; j < 3; ++j) {
                        pos[j] = __half2float(m[j]) * pos_world[0] +
                                 __half2float(m[3 + j]) * pos_world[1] +
                                 __half2float(m[6 + j]) * pos_world[2] +
                                 __half2float(m[9 + j]);
                        pos[j] = tree.offset[j] + tree.scale[j] * pos[j];
                    }
                    sigma = __half2float(warp.max_sigma);

                } else {
                    const half* m = warp.transform;
                    const uint32_t c000 = morton_code_3(coords[0]+1, coords[1]-1, coords[2]-1);
                    const uint32_t c001 = morton_code_3(coords[0]+1, coords[1]-1, coords[2]+1);
                    const uint32_t c010 = morton_code_3(coords[0]-1, coords[1]-1, coords[2]-1);
                    const uint32_t c011 = morton_code_3(coords[0]-1, coords[1]-1, coords[2]+1);
                    const uint32_t c100 = morton_code_3(coords[0]+1, coords[1]+1, coords[2]-1);
                    const uint32_t c101 = morton_code_3(coords[0]+1, coords[1]+1, coords[2]+1);
                    const uint32_t c110 = morton_code_3(coords[0]-1, coords[1]+1, coords[2]-1);
                    const uint32_t c111 = morton_code_3(coords[0]-1, coords[1]+1, coords[2]+1);

                    uint32_t neighbors[] = {c000, c001, c010, c011, c100, c101, c110, c111};
                    float avg_transform[12];
                    for (int k = 0; k < 12; k++) {
                        avg_transform[k] = 0;
                        for (int n = 0; n < 8; n++) {
                            avg_transform[k] += __half2float(tree.warp[neighbors[n]].transform[k]);
                        }
                            avg_transform[k] = avg_transform[k] / 1024 + __half2float(m[k]) *127.0 / 128.0;
                   }

                    for (int j = 0; j < 3; ++j) {
                        pos[j] = avg_transform[j] * pos_world[0] +
                                avg_transform[3 + j] * pos_world[1] +
                                avg_transform[6 + j] * pos_world[2] +
                                avg_transform[9 + j];
                        pos[j] = tree.offset[j] + tree.scale[j] * pos[j];
                    }
                    sigma = __half2float(warp.max_sigma);
                }
            }

            // Now pos is the position in canonical space, query the octree
            query_single_from_root(tree, pos, &tree_val, &cube_sz);

            sigma = min(sigma, __half2float(tree_val[tree.data_dim - 1]));
            // Volume rendering step
            if (sigma > opt.sigma_thresh) {
                scalar_t att = expf(-delta_t * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                if (opt.render_depth) {
                    out[0] += weight * t;
                } else {
                    out[0] += __half2float(tree_val[0]) * weight;
                    out[1] += __half2float(tree_val[1]) * weight;
                    out[2] += __half2float(tree_val[2]) * weight;
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
