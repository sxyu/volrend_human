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

                scalar_t x = pos[0] - 0.5;
                scalar_t y = pos[1] - 0.5;
                scalar_t z = pos[2] - 0.5;

                uint32_t x0 = floorf(x);
                uint32_t x1 = x0 + 1;
                uint32_t y0 = floorf(y);
                uint32_t y1 = y0 + 1;
                uint32_t z0 = floorf(z);
                uint32_t z1 = z0 + 1;

//                printf("x: %f  y: %f  z: %f  x0: %d  y0: %d  z0: %d\n", x, y, z, x0, y0, z0);

                if (x0 < 0 || y0 < 0 || z0 < 0 ||
                    x1 >= N3_WARP_GRID_SIZE || y1 >= N3_WARP_GRID_SIZE ||
                    z1 >= N3_WARP_GRID_SIZE) {
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

                    const uint32_t c000 = morton_code_3(x1, y0, z0);
                    const uint32_t c001 = morton_code_3(x1, y0, z1);
                    const uint32_t c010 = morton_code_3(x0, y0, z0);
                    const uint32_t c011 = morton_code_3(x0, y0, z1);
                    const uint32_t c100 = morton_code_3(x1, y1, z0);
                    const uint32_t c101 = morton_code_3(x1, y1, z1);
                    const uint32_t c110 = morton_code_3(x0, y1, z0);
                    const uint32_t c111 = morton_code_3(x0, y1, z1);

                    float xd = x - x0;
                    float yd = y - y0;
                    float zd = z - z0;

                    float interpolated_transform[12];
                    for (int k = 0; k < 12; k++) {
                        interpolated_transform[k] = 0;
                        float v000 = __half2float(tree.warp[c000].transform[k]);
                        float v001 = __half2float(tree.warp[c001].transform[k]);
                        float v010 = __half2float(tree.warp[c010].transform[k]);
                        float v011 = __half2float(tree.warp[c011].transform[k]);
                        float v100 = __half2float(tree.warp[c100].transform[k]);
                        float v101 = __half2float(tree.warp[c101].transform[k]);
                        float v110 = __half2float(tree.warp[c110].transform[k]);
                        float v111 = __half2float(tree.warp[c111].transform[k]);

                        float v00 = v000 * (1 - xd) + v100 * xd;
                        float v01 = v001 * (1 - xd) + v101 * xd;
                        float v10 = v010 * (1 - xd) + v110 * xd;
                        float v11 = v011 * (1 - xd) + v111 * xd;

                        float v0 = v00 * (1 - yd) + v10 * yd;
                        float v1 = v01 * (1 - yd) + v11 * yd;

                        float v = v0 * (1 - zd) + v1 * zd;

                        interpolated_transform[k] = v;
                    }
                    for (int j = 0; j < 3; ++j) {
                        pos[j] = __half2float(interpolated_transform[j]) * pos_world[0] +
                                 __half2float(interpolated_transform[3 + j]) * pos_world[1] +
                                 __half2float(interpolated_transform[6 + j]) * pos_world[2] +
                                 __half2float(interpolated_transform[9 + j]);
                        pos[j] = tree.offset[j] + tree.scale[j] * pos[j];
                    }
                    float s000 = __half2float(tree.warp[c000].max_sigma);
                    float s001 = __half2float(tree.warp[c001].max_sigma);
                    float s010 = __half2float(tree.warp[c010].max_sigma);
                    float s011 = __half2float(tree.warp[c011].max_sigma);
                    float s100 = __half2float(tree.warp[c100].max_sigma);
                    float s101 = __half2float(tree.warp[c101].max_sigma);
                    float s110 = __half2float(tree.warp[c110].max_sigma);
                    float s111 = __half2float(tree.warp[c111].max_sigma);

                    float s00 = s000 * (1 - xd) + s100 * xd;
                    float s01 = s001 * (1 - xd) + s101 * xd;
                    float s10 = s010 * (1 - xd) + s110 * xd;
                    float s11 = s011 * (1 - xd) + s111 * xd;

                    float s0 = s00 * (1 - yd) + s10 * yd;
                    float s1 = s01 * (1 - yd) + s11 * yd;

                    float s = s0 * (1 - zd) + s1 * zd;
//                    printf("xd: %f  yd: %f  zd: %f\n", xd, yd, zd);
//                    printf("original: %f  s:%f   s000: %f  s001: %f  s010: %f  s011: %f  s100: %f  s101: %f  s110: %f  s111: %f  \n", __half2float(warp.max_sigma), s, s000, s001, s010, s011, s100, s101, s110, s111);
//                    float arr[] = {s000, s001, s010, s011, s100, s101, s110, s111};
//                    float max = s000;
//                    for (int i = 0; i < 8; i++){
//                        if (arr[i] > max){
//                            max = arr[i];
//                        }
//                    }
//                    sigma = max ;
                    sigma = s; // interpolated max_sigma
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
