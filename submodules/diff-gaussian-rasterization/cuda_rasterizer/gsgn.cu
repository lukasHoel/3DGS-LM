#include "gsgn_data_spec.h"
#include "gsgn.h"
#include "rasterizer_impl.h"
#include "auxiliary.h"
#include "hip_warp_compat.h"
#include <cooperative_groups.h>
#include <iostream>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#if defined(USE_ROCM)
// torch's hipify maps the <cub/...> include to hipCUB and rewrites most cub::
// symbols, but it misses some (e.g. cub::WarpScan inside a `using` alias). The
// namespace alias makes every cub:: reference resolve regardless.
namespace cub = hipcub;
#endif

namespace cg = cooperative_groups;

#define SPARSE_J_NUM_THREADS 128
#define SPARSE_JT_NUM_THREADS 128
// FULL_MASK is defined in hip_warp_compat.h: 64-bit on HIP (the *_sync builtins
// static_assert sizeof(mask) == 8) and 0xffffffff on CUDA.

namespace CudaRasterizer {
    namespace GSGN {

        // template magic part 1: 
        // - each specialization needs different variables.
        // - we want to avoid allocating variables that are not needed (reduce register count).
        // - use specialized templated structs that only contain the necessary variables
        template <uint32_t C, typename scalar_t, GSGN_MODE M> struct KernelVars;
        template <uint32_t C, typename scalar_t> struct KernelVars<C, scalar_t, GSGN_MODE::EVAL_JTF_AND_SPARSE_INTERMEDIATE> {
            float residual[C];
            int32_t offset;
            int32_t stride;
            GaussianCache cache;
        };
        template <uint32_t C, typename scalar_t> struct KernelVars<C, scalar_t, GSGN_MODE::APPLY_JTJ> {
            scalar_t jx[C] = { 0 };
        };
        template <uint32_t C, typename scalar_t> struct KernelVars<C, scalar_t, GSGN_MODE::PRECONDITIONER> {
            // does not need extra variables
        };
        template <uint32_t C, typename scalar_t> struct KernelVars<C, scalar_t, GSGN_MODE::APPLY_J> {
            scalar_t jx[C] = { 0 };
        };

        enum class GAUSSIAN_ATTRIBUTE {
            POSITION,
            SCALE,
            OPACITY,
            ROTATION,
            FEAT_DC,
            FEAT_REST
        };

        template <GAUSSIAN_ATTRIBUTE P>
        __device__ __forceinline__ int32_t get_vector_position(const int32_t global_id, const int32_t channel, const uint32_t idx, PackedGSGNDataSpec& data) {
            // global_id: the global_id of the gaussian
            // channel: which color channel we want to query (vector stores gradients w.r.t. residuals and each color channel is one residual)
            // idx: if a gaussian attribute has multiple values (e.g. the position has 3 values: x, y, z), idx specifies which one

            if constexpr(P == GAUSSIAN_ATTRIBUTE::POSITION) {
                assert(idx < 3);
                // is independent of channel: each channel contributes to all gradient attributes
                // layouted like this: [px0, ..., pxN, py0, ..., pyN, pz0, ..., pzN]
                return data.offset_xyz + global_id + idx * data.P;
            } else if constexpr(P == GAUSSIAN_ATTRIBUTE::SCALE) {
                assert(idx < 3);
                // is independent of channel: each channel contributes to all gradient attributes
                // layouted like this: [sx0, ..., sxN, sy0, ..., syN, sz0, ..., szN]
                return data.offset_scales + global_id + idx * data.P;
            } else if constexpr(P == GAUSSIAN_ATTRIBUTE::ROTATION) {
                assert(idx < 4);
                // is independent of channel: each channel contributes to all gradient attributes
                // layouted like this: [rx0, ..., rxN, ry0, ..., ryN, rz0, ..., rzN, rw0, ..., rwN]
                return data.offset_rotations + global_id + idx * data.P;
            } else if constexpr(P == GAUSSIAN_ATTRIBUTE::OPACITY) {
                assert(idx == 0); // opacity only is a single value
                // is independent of channel: each channel contributes to all gradient attributes
                // layouted like this: [o0, ..., oN]
                return data.offset_opacity + global_id;
            } else if constexpr(P == GAUSSIAN_ATTRIBUTE::FEAT_DC) {
                assert(channel < 3);
                // each color channel only gets gradients for the respective channel-th SH attributes.
                // layouted like this: [r0, ..., rN, g0, ..., gN, b0, ..., bN] where r0 could also be named sh0_r0, consistent with naming in FEAT_REST section
                return data.offset_features_dc + global_id + channel * data.P;
            } else if constexpr(P == GAUSSIAN_ATTRIBUTE::FEAT_REST) {
                assert(channel < 3 && idx < (data.M - 1));
                // each color channel only gets gradients for the respective channel-th SH attributes.
                // layouted like this: [sh1_r0, ..., sh1_rN, sh1_g0, ..., sh1_gN, sh1_b0, ..., sh1_bN, ..., shM_r0, ..., shM_rN, shM_g0, ..., shM_gN, shM_b0, ..., shM_bN]
                return data.offset_features_rest + global_id + (3 * idx + channel) * data.P;
            }
            return -1;
        }

        // DISTWAR - serialized atomic reduction (SW-S)
        // Leader election over the lanes updating the same primitive. The
        // recombination is via atomicAdd, so warp granularity is flexible: on
        // HIP this spans the full hardware wavefront (a 64-bit same-primitive
        // mask), which is correct and uses fewer atomics. __match_any_sync
        // partitions lanes by value, so two distinct primitives in one wavefront
        // each elect their own leader. laneId/sync_mask are wavefront-relative.
        template<typename ATOM_T>
        __device__ void atomred_vec(unsigned int laneId, size_t idx, ATOM_T** ptr, ATOM_T *val, size_t len, unsigned int balance_threshold) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
            laneId = gsgn_wave_lane();
#endif
            // a mask of threads in the warp updating the same primitive
            GSGN_LANE_MASK same_mask = __match_any_sync(__activemask(), idx);

            // number of threads in the warp updating the same primitive
            unsigned same_ct = gsgn_popc(same_mask);

            /* if number of threads updating current
            primitive exceeds balance threshold, perform
            serialized warp level reduction */
            if (same_ct >= balance_threshold) {
                // thread with lowest id becomes the leader
                unsigned leader = gsgn_ffs(same_mask) - 1;

                // leader does not fetch from itself
                same_mask &= ~gsgn_lane_bit(leader);

                /* leader fetch and accumulate all parameters
                from threads updating the same primitive */
                while (same_mask) {
                    unsigned target_lane = gsgn_ffs(same_mask) - 1;
                    if (laneId == leader || laneId == target_lane) {
                        GSGN_LANE_MASK sync_mask = gsgn_lane_bit(leader) | gsgn_lane_bit(target_lane);

                        for (unsigned i = 0; i < len; ++i) {
                            val[i] += __shfl_sync(sync_mask, val[i], target_lane);
                        }
                    }

                    same_mask &= ~gsgn_lane_bit(target_lane);
                }

                // leader sends an atomicAdd per parameter
                if (laneId == leader) {
                    for (unsigned i = 0; i < len; ++i) {
                        atomicAdd(ptr[i], val[i]);
                    }
                }
            } else { // balance threshold not met, update normally
                for (unsigned i = 0; i < len; ++i) {
                    atomicAdd(ptr[i], val[i]);
                }
            }
        }

        template <typename scalar_t>
        __global__ void __launch_bounds__(GSGN_BLOCK_X * GSGN_BLOCK_Y)
        eval_jtf_sparse_render_bkwd_kernel(
            PackedGSGNDataSpec data,
            const int img_id,
            const int dL_offset,
            // output variables
            scalar_t* __restrict__ r_vec,
            scalar_t* __restrict__ dL_dcolors,
            scalar_t* __restrict__ dL_dmean2D,
            scalar_t* __restrict__ dL_dconic2D,
            __half* __restrict__ sparse_jacobians,
            int* __restrict__ index_map) {

            // this kernel computes the first part of JT * F (similar to SGD) for a single image
            // each thread handles one ray and loops over all gaussians along that ray
            // we write out the values by atomically adding to the output position (but with warp-reduction)

            auto block = cg::this_thread_block();
            const uint32_t horizontal_blocks = (data.W + GSGN_BLOCK_X - 1) / GSGN_BLOCK_X;
            const uint32_t block_idx = block.group_index().y * horizontal_blocks + block.group_index().x;
            const uint32_t rank = block.thread_rank();
            const uint32_t lane_id = rank % 32;
            const uint32_t warp_id = rank / 32;
            constexpr uint32_t NUM_WARPS = GSGN_BLOCK_X * GSGN_BLOCK_Y / 32;

            const uint2 range = data.ranges_ptrs[img_id][block_idx];
            const int rounds = (range.y - range.x + 31) / 32;

            // We rasterize again. Compute necessary block info.
            const uint2 pix_min = { block.group_index().x * GSGN_BLOCK_X, block.group_index().y * GSGN_BLOCK_Y };
            uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
            const bool inside = pix.x < data.W && pix.y < data.H;
            uint32_t pix_id = data.W * pix.y + pix.x;
            const float2 pixf = { (float)pix.x, (float)pix.y };

            __shared__ int collected_id[NUM_WARPS][32];
            __shared__ char rendered_cache[NUM_WARPS][sizeof(GeometryStateReduced)];
            __shared__ float unactivated_opacity[NUM_WARPS];
            
            // Pin the logical warp width to 32. hipCUB defaults LOGICAL_WARP_THREADS
            // to warpSize (64 on gfx90a); the surrounding code partitions the block
            // into 32-lane warps with temp_storage[NUM_WARPS] and a [NUM_WARPS][32]
            // layout, so an unpinned 64-wide scan would fold two logical warps into
            // one TempStorage slot and corrupt the per-warp ExclusiveSum offset.
            using WarpScan = cub::WarpScan<int, GSGN_LOGICAL_WARP_SIZE>;
            __shared__ typename WarpScan::TempStorage temp_storage[NUM_WARPS];

            // In the forward, we stored the final value for T, the
            // product of all (1 - alpha) factors.
            const float T_final = inside ? data.accum_alpha_ptrs[img_id][pix_id] : 0;

            // We start from the back. The ID of the last contributing
            // Gaussian is known from each pixel from the forward.
            const int last_contributor = inside ? data.n_contrib_ptrs[img_id][pix_id] : 0;

            float residual[GSGN_NUM_CHANNELS];
            if (inside) {
                #pragma unroll
                for (int i = 0; i < GSGN_NUM_CHANNELS; i++) {
                    uint32_t pos = i * data.jx_stride + img_id * data.num_pixels + pix_id;
                    residual[i] = data.residuals[pos];
                }
            }
            
            const uint32_t global_warp_id = block.group_index().x + block.group_index().y * horizontal_blocks * NUM_WARPS + warp_id * horizontal_blocks;
            uint32_t offset = inside ? data.n_contrib_vol_rend_prefix_sum[img_id][global_warp_id] : 0;
            const uint32_t stride = inside ? data.n_sparse_gaussians[img_id] : 0;

            float T_curr = T_final;
            int toDo = range.y - range.x;
            int contributor = toDo;
            float last_alpha = 0;
            float last_color[GSGN_NUM_CHANNELS] = { 0 };
            float accum_rec[GSGN_NUM_CHANNELS] = { 0 };

            // Traverse all Gaussians
            for (int i = 0; i < rounds; i++, toDo -= 32) {
                // Load auxiliary data into shared memory, start in the BACK
                // and load them in reverse order.

                __syncwarp();
                const int progress = i * 32 + lane_id;
                if (range.x + progress < range.y) {
                    const int coll_id = data.point_list_ptrs[img_id][range.y - progress - 1];
                    collected_id[warp_id][lane_id] = coll_id;
                }
                __syncwarp();

                // Iterate over Gaussians
                for (int j = 0; j < min(32, toDo); j++) {

                    contributor--; // Keep track of current Gaussian ID.
                    int global_id = collected_id[warp_id][j];
                    int dL_pos = data.map_visible_gaussians[img_id][global_id];

                    if(dL_pos == -1) {
                        // can continue to next gaussian if it is not visible in any pixel (marked as -1 in the map)
                        // also need to skip it because otherwise the power/alpha calculations below are wrong since the rendered_data_ptr contains invalid data
                        // every thread in the warp will process this gaussian at the same j-th loop iteration, so every thread will also skip it
                        // the subsequent __syncwarp() statements will thus still work
                        continue;
                    }

                    // load next data into shared memory, collaboratively for this warp
                    __syncwarp();
                    GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + dL_pos;
                    uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                    reinterpret_cast<uint16_t*>(rendered_cache[warp_id])[lane_id] = rendered_read_in_ptr[lane_id];
                    rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(rendered_cache[warp_id]);
                    unactivated_opacity[warp_id] = data.unactivated_opacity[global_id];
                    __syncwarp();

                    // Skip, if this gaussian is behind the last contributor for this pixel.
                    int is_active = inside && (contributor < last_contributor);
                    
                    // Compute blending values, as before.
                    float2 d = { rendered_data_ptr->means2D[0] - pixf.x, rendered_data_ptr->means2D[1] - pixf.y };
                    float power = -0.5f * (rendered_data_ptr->conic_opacity[0] * d.x * d.x + rendered_data_ptr->conic_opacity[2] * d.y * d.y) - rendered_data_ptr->conic_opacity[1] * d.x * d.y;
                    is_active = is_active && power <= 0.0f;
                    
                    float G = exp(power);
                    float alpha = min(0.99f, rendered_data_ptr->conic_opacity[3] * G);
                    is_active = is_active && alpha >= GSGN_ALPHA_THRESH;

                    bool skip = is_active == 0;

                    // scan how many warps are active, this gives the position to write to for this thread
                    int n_active_threads;
                    WarpScan(temp_storage[warp_id]).ExclusiveSum(is_active, is_active, n_active_threads);
                    int sparse_jac_pos = offset + is_active;
                    offset += n_active_threads;

                    // now can skip the rest for this thread
                    if(skip) {
                        continue;
                    }

                    T_curr = T_curr / (1.f - alpha);
                    const float dchannel_dcolor = alpha * T_curr;

                    // write to index_map
                    index_map[sparse_jac_pos] = global_id;
                    index_map[sparse_jac_pos + stride] = data.num_pixels * img_id + pix_id;

                    // construct sparse jacobian entry (will be written out all at once)
                    GradientCache sparse_jac_entry;
                    sparse_jac_entry.dchannel_dcolor = __float2half(dchannel_dcolor);

                    // Propagate gradients to per-Gaussian colors and keep
                    // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
                    // pair).
                    float dL_dalpha_residual = 0.0f;

                    scalar_t *atom_ptrs[9];
                    scalar_t atom_vals[9];

                    #pragma unroll
                    for (int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                        // Update last color (to be used in the next iteration)
                        accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                        last_color[ch] = rendered_data_ptr->rgb[ch];

                        // get the gradient w.r.t. alpha of the Gaussian.
                        float dL_dalpha = (rendered_data_ptr->rgb[ch] - accum_rec[ch]);
                        dL_dalpha *= T_curr;

                        // get the gradient w.r.t. ch-th color channel of the Gaussian.
                        atom_ptrs[ch] = &dL_dcolors[dL_pos + ch * dL_offset];
                        atom_vals[ch] = dchannel_dcolor * residual[ch];

                        // Account for fact that alpha also influences how much of
                        // the background color is added if nothing left to blend
                        float bg_dot_dpixel = data.background[ch];
                        dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

                        sparse_jac_entry.dL_dalpha[ch] = __float2half(dL_dalpha);
                        dL_dalpha_residual += dL_dalpha * residual[ch]; // grad * residual for b = J^T * F
                    }

                    // write to sparse_jacobian in one go (8 Byte per thread --> a single 256 Byte write instruction)
                    reinterpret_cast<GradientCache*>(sparse_jacobians)[sparse_jac_pos] = sparse_jac_entry;

                    // Update last alpha (to be used in the next iteration)
                    last_alpha = alpha;

                    // Helpful reusable temporary variables
                    const float dalpha_dG = rendered_data_ptr->conic_opacity[3];
                    const float gdx = G * d.x;
                    const float gdy = G * d.y;
                    const float dG_ddelx = -gdx * rendered_data_ptr->conic_opacity[0] - gdy * rendered_data_ptr->conic_opacity[1];
                    const float dG_ddely = -gdy * rendered_data_ptr->conic_opacity[2] - gdx * rendered_data_ptr->conic_opacity[1];

                    // Gradient of pixel coordinate w.r.t. normalized
                    // screen-space viewport corrdinates (-1 to 1)
                    const float ddelx_dx = 0.5f * data.W;
                    const float ddely_dy = 0.5f * data.H;

                    // get gradients w.r.t. 2D mean position of the Gaussian
                    float2 dalpha_dmean2D;
                    dalpha_dmean2D.x = dalpha_dG * dG_ddelx * ddelx_dx;
                    dalpha_dmean2D.y = dalpha_dG * dG_ddely * ddely_dy;

                    atom_ptrs[3] = &dL_dmean2D[dL_pos + 0 * dL_offset];
                    atom_vals[3] = dalpha_dmean2D.x * dL_dalpha_residual;

                    atom_ptrs[4] = &dL_dmean2D[dL_pos + 1 * dL_offset];
                    atom_vals[4] = dalpha_dmean2D.y * dL_dalpha_residual;

                    // get gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
                    float3 dalpha_dconic2D;
                    dalpha_dconic2D.x = -0.5f * gdx * d.x * dalpha_dG;
                    dalpha_dconic2D.y = -0.5f * gdx * d.y * dalpha_dG;
                    dalpha_dconic2D.z = -0.5f * gdy * d.y * dalpha_dG;
                    
                    atom_ptrs[5] = &dL_dconic2D[dL_pos + 0 * dL_offset];
                    atom_vals[5] = dalpha_dconic2D.x * dL_dalpha_residual;

                    atom_ptrs[6] = &dL_dconic2D[dL_pos + 1 * dL_offset];
                    atom_vals[6] = dalpha_dconic2D.y * dL_dalpha_residual;

                    atom_ptrs[7] = &dL_dconic2D[dL_pos + 2 * dL_offset];
                    atom_vals[7] = dalpha_dconic2D.z * dL_dalpha_residual;

                    // get gradients w.r.t. opacity of the Gaussian
                    scalar_t dalpha_dopacity = G;
                    dalpha_dopacity = (scalar_t) dsigmoidvdv((scalar_t) unactivated_opacity[warp_id], dalpha_dopacity);
                    int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::OPACITY>(global_id, 0, 0, data);

                    atom_ptrs[8] = &r_vec[pos_out];
                    atom_vals[8] = dalpha_dopacity * dL_dalpha_residual;

                    // DISTWAR - perform all atomic updates using serialized atomic reduction (SW-S)
			        atomred_vec(lane_id, global_id, atom_ptrs, atom_vals, 9, 1);
                }
            }
        }

        // Backward version of INVERSE 2D covariance matrix computation
        // (due to length launched as separate kernel before other 
        // backward steps contained in preprocess)
        template <typename scalar_t, bool write_cache>
        __global__ void __launch_bounds__(256)
        gsgn_computeCov2DCUDA(
            PackedGSGNDataSpec data,
            const int img_id,
            const int n_visible_gaussians,
            const scalar_t* dL_dconics,
            scalar_t* out_vec,
            scalar_t* dL_dcov,
            float* per_gaussian_cache,
            int* map_cache_to_gaussians) {

            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

            constexpr uint32_t NUM_ATTRS = sizeof(GaussianCacheComputeCov2D) / sizeof(float);
            __shared__ float cache[256][NUM_ATTRS];
            
            if constexpr(! write_cache) {
                // collaboratively read in from cache
                GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache) + blockIdx.x * blockDim.x;
                #pragma unroll
                for(int i = 0; i < NUM_ATTRS; i++) {
                    const uint32_t cache_idx = threadIdx.x + i * 256;
                    const uint32_t tid = cache_idx / NUM_ATTRS;
                    const uint32_t attr_id = cache_idx % NUM_ATTRS;
                    if(tid + blockIdx.x * blockDim.x >= n_visible_gaussians) break;
                    float* linearized_cache_ptr = reinterpret_cast<float*>(cache_ptr + tid);
                    cache[tid][attr_id] = linearized_cache_ptr[attr_id];
                }
                __syncthreads();
            }

            if (idx < n_visible_gaussians) {
                const int global_id = map_cache_to_gaussians[idx];
                GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + idx;
                // Don't have to check because we only parallelize over n_visible_gaussians -- all of those have a radius_gt_zero
                // if(! rendered_data_ptr->radius_gt_zero) {
                //     return;
                // }

                // TODO: could live in shared memory
                const float* view_matrix = data.viewmatrix[img_id];

                GaussianCacheComputeCov2D* cache_ptr = reinterpret_cast<GaussianCacheComputeCov2D*>(cache[threadIdx.x]);
                if constexpr(write_cache) {
                    // write to per_gaussian_cache
                    float3 mean = data.means3D[global_id];
                    float3 t = {
                        view_matrix[0] * mean.x + view_matrix[4] * mean.y + view_matrix[ 8] * mean.z + view_matrix[12],
                        view_matrix[1] * mean.x + view_matrix[5] * mean.y + view_matrix[ 9] * mean.z + view_matrix[13],
                        view_matrix[2] * mean.x + view_matrix[6] * mean.y + view_matrix[10] * mean.z + view_matrix[14],
                    };

                    const float h_x = data.focal_x[img_id];
                    const float h_y = data.focal_y[img_id];
                    const float tan_fovx = data.tan_fovx[img_id];
                    const float tan_fovy = data.tan_fovy[img_id];

                    // original
                    const float limx = 1.3f * tan_fovx;
                    const float limy = 1.3f * tan_fovy;
                    const float txtz = t.x / t.z;
                    const float tytz = t.y / t.z;
                    t.x = min(limx, max(-limx, txtz)) * t.z;
                    t.y = min(limy, max(-limy, tytz)) * t.z;
                    const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
                    const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

                    // fix inspired by https://github.com/nerfstudio-project/gsplat/pull/305/files
                    // const float cx = data.cx[img_id];
                    // const float cy = data.cy[img_id];
                    // const float lim_x_pos = (data.W - cx) / h_x + 0.3f * tan_fovx;
                    // const float lim_x_neg = cx / h_x + 0.3f * tan_fovx;
                    // const float lim_y_pos = (data.H - cy) / h_y + 0.3f * tan_fovy;
                    // const float lim_y_neg = cy / h_y + 0.3f * tan_fovy;
                    // t.x = min(lim_x_pos, max(-lim_x_neg, txtz)) * t.z;
                    // t.y = min(lim_y_pos, max(-lim_y_neg, tytz)) * t.z;
                    // const float x_grad_mul = txtz < -lim_x_neg || txtz > lim_x_pos ? 0 : 1;
	                // const float y_grad_mul = tytz < -lim_y_neg || tytz > lim_y_pos ? 0 : 1;

                    float T[6] = {
                        view_matrix[0] * h_x / t.z - view_matrix[ 2] * (h_x * t.x) / (t.z * t.z),
                        view_matrix[4] * h_x / t.z - view_matrix[ 6] * (h_x * t.x) / (t.z * t.z),
                        view_matrix[8] * h_x / t.z - view_matrix[10] * (h_x * t.x) / (t.z * t.z),
                        view_matrix[1] * h_y / t.z - view_matrix[ 2] * (h_y * t.y) / (t.z * t.z),
                        view_matrix[5] * h_y / t.z - view_matrix[ 6] * (h_y * t.y) / (t.z * t.z),
                        view_matrix[9] * h_y / t.z - view_matrix[10] * (h_y * t.y) / (t.z * t.z)
                    };

                    // Reading location of 3D covariance for this Gaussian
                    const float* cov3D = rendered_data_ptr->cov3D;

                    // Use helper variables for 2D covariance entries. More compact.
                    float a = 0.3f + T[0] * (cov3D[0] * T[0] + cov3D[1] * T[1] + cov3D[2] * T[2]) + T[1] * (cov3D[1] * T[0] + cov3D[3] * T[1] + cov3D[4] * T[2]) + T[2] * (cov3D[2] * T[0] + cov3D[4] * T[1] + cov3D[5] * T[2]);
                    float b = T[3] * (cov3D[0] * T[0] + cov3D[1] * T[1] + cov3D[2] * T[2]) + T[4] * (cov3D[1] * T[0] + cov3D[3] * T[1] + cov3D[4] * T[2]) + T[5] * (cov3D[2] * T[0] + cov3D[4] * T[1] + cov3D[5] * T[2]);
                    float c = 0.3f + T[3] * (cov3D[0] * T[3] + cov3D[1] * T[4] + cov3D[2] * T[5]) + T[4] * (cov3D[1] * T[3] + cov3D[3] * T[4] + cov3D[4] * T[5]) + T[5] * (cov3D[2] * T[3] + cov3D[4] * T[4] + cov3D[5] * T[5]);

                    float tmp1 = (T[0] * cov3D[0] + T[1] * cov3D[1] + T[2] * cov3D[2]);
                    float tmp2 = (T[3] * cov3D[0] + T[4] * cov3D[1] + T[5] * cov3D[2]);
                    float tmp3 = (T[0] * cov3D[1] + T[1] * cov3D[3] + T[2] * cov3D[4]);
                    float tmp4 = (T[3] * cov3D[1] + T[4] * cov3D[3] + T[5] * cov3D[4]);
                    float tmp5 = (T[0] * cov3D[2] + T[1] * cov3D[4] + T[2] * cov3D[5]);
                    float tmp6 = (T[3] * cov3D[2] + T[4] * cov3D[4] + T[5] * cov3D[5]);

                    float denom = a * c - b * b;
                    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

                    float tz = 1.f / t.z;
                    float tz2 = tz * tz;
                    float tz3 = tz2 * tz;

                    cache_ptr->t[0] = 2 * h_x * t.x * tz3; // we never need t.x alone, only this product is used --> precompute it completely
                    cache_ptr->t[1] = 2 * h_y * t.y * tz3; // we never need t.y alone, only this product is used --> precompute it completely
                    cache_ptr->t[2] = -h_x * tz2; // we never need t.z alone, only this product is used --> precompute it completely
                    cache_ptr->t[3] = -h_y * tz2; // we never need t.z alone, only this product is used --> precompute it completely
                    cache_ptr->T[0] = T[0];
                    cache_ptr->T[1] = T[1];
                    cache_ptr->T[2] = T[2];
                    cache_ptr->T[3] = T[3];
                    cache_ptr->T[4] = T[4];
                    cache_ptr->T[5] = T[5];
                    cache_ptr->a_ = a;
                    cache_ptr->b_ = b;
                    cache_ptr->c_ = c;
                    cache_ptr->x_grad_mul = x_grad_mul;
                    cache_ptr->y_grad_mul = y_grad_mul;
                    cache_ptr->dL_dT_precomp[0] = tmp1;
                    cache_ptr->dL_dT_precomp[1] = tmp2;
                    cache_ptr->dL_dT_precomp[2] = tmp3;
                    cache_ptr->dL_dT_precomp[3] = tmp4;
                    cache_ptr->dL_dT_precomp[4] = tmp5;
                    cache_ptr->dL_dT_precomp[5] = tmp6;
                    cache_ptr->denom = denom;
                    cache_ptr->denom2inv = denom2inv;
                }

                float dL_da = 0, dL_db = 0, dL_dc = 0;
                float3 dL_dconic_residual = { (float) dL_dconics[idx], (float) dL_dconics[idx + n_visible_gaussians], (float) dL_dconics[idx + 2 * n_visible_gaussians] };

                if (cache_ptr->denom2inv != 0) {
                    // Gradients of loss w.r.t. entries of 2D covariance matrix,
                    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
                    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a

                    dL_da = cache_ptr->denom2inv * (-cache_ptr->c_ * cache_ptr->c_ * dL_dconic_residual.x + 2 * cache_ptr->b_ * cache_ptr->c_ * dL_dconic_residual.y + (cache_ptr->denom - cache_ptr->a_ * cache_ptr->c_) * dL_dconic_residual.z);
                    dL_dc = cache_ptr->denom2inv * (-cache_ptr->a_ * cache_ptr->a_ * dL_dconic_residual.z + 2 * cache_ptr->a_ * cache_ptr->b_ * dL_dconic_residual.y + (cache_ptr->denom - cache_ptr->a_ * cache_ptr->c_) * dL_dconic_residual.x);
                    dL_db = cache_ptr->denom2inv * 2 * (cache_ptr->b_ * cache_ptr->c_ * dL_dconic_residual.x - (cache_ptr->denom + 2 * cache_ptr->b_ * cache_ptr->b_) * dL_dconic_residual.y + cache_ptr->a_ * cache_ptr->b_ * dL_dconic_residual.z);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (diagonal).
                    // cov2D = transpose(T) * transpose(Vrk) * T;

                    dL_dcov[idx] = (cache_ptr->T[0] * cache_ptr->T[0] * dL_da + cache_ptr->T[0] * cache_ptr->T[3] * dL_db + cache_ptr->T[3] * cache_ptr->T[3] * dL_dc);
                    dL_dcov[idx + 3 * n_visible_gaussians] = (cache_ptr->T[1] * cache_ptr->T[1] * dL_da + cache_ptr->T[1] * cache_ptr->T[4] * dL_db + cache_ptr->T[4] * cache_ptr->T[4] * dL_dc);
                    dL_dcov[idx + 5 * n_visible_gaussians] = (cache_ptr->T[2] * cache_ptr->T[2] * dL_da + cache_ptr->T[2] * cache_ptr->T[5] * dL_db + cache_ptr->T[5] * cache_ptr->T[5] * dL_dc);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
                    // Off-diagonal elements appear twice --> double the gradient.
                    // cov2D = transpose(T) * transpose(Vrk) * T;

                    dL_dcov[idx + 1 * n_visible_gaussians] = 2 * cache_ptr->T[0] * cache_ptr->T[1] * dL_da + (cache_ptr->T[0] * cache_ptr->T[4] + cache_ptr->T[1] * cache_ptr->T[3]) * dL_db + 2 * cache_ptr->T[3] * cache_ptr->T[4] * dL_dc;
                    dL_dcov[idx + 2 * n_visible_gaussians] = 2 * cache_ptr->T[0] * cache_ptr->T[2] * dL_da + (cache_ptr->T[0] * cache_ptr->T[5] + cache_ptr->T[2] * cache_ptr->T[3]) * dL_db + 2 * cache_ptr->T[3] * cache_ptr->T[5] * dL_dc;
                    dL_dcov[idx + 4 * n_visible_gaussians] = 2 * cache_ptr->T[2] * cache_ptr->T[1] * dL_da + (cache_ptr->T[1] * cache_ptr->T[5] + cache_ptr->T[2] * cache_ptr->T[4]) * dL_db + 2 * cache_ptr->T[4] * cache_ptr->T[5] * dL_dc;
                } else {
                    #pragma unroll
                    for (int i = 0; i < 6; i++) {
                        dL_dcov[idx + i * n_visible_gaussians] = 0;
                    }
                }

                // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
                // cov2D = transpose(T) * transpose(Vrk) * T;
                float dL_dT00 = 2 * cache_ptr->dL_dT_precomp[0] * dL_da + cache_ptr->dL_dT_precomp[1] * dL_db;
                float dL_dT10 = 2 * cache_ptr->dL_dT_precomp[1] * dL_dc + cache_ptr->dL_dT_precomp[0] * dL_db;
                float dL_dT01 = 2 * cache_ptr->dL_dT_precomp[2] * dL_da + cache_ptr->dL_dT_precomp[3] * dL_db;
                float dL_dT11 = 2 * cache_ptr->dL_dT_precomp[3] * dL_dc + cache_ptr->dL_dT_precomp[2] * dL_db;
                float dL_dT02 = 2 * cache_ptr->dL_dT_precomp[4] * dL_da + cache_ptr->dL_dT_precomp[5] * dL_db;
                float dL_dT12 = 2 * cache_ptr->dL_dT_precomp[5] * dL_dc + cache_ptr->dL_dT_precomp[4] * dL_db;

                // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
                // T = W * J
                float dL_dJ00 = view_matrix[0] * dL_dT00 + view_matrix[4] * dL_dT01 + view_matrix[ 8] * dL_dT02;
                float dL_dJ02 = view_matrix[2] * dL_dT00 + view_matrix[6] * dL_dT01 + view_matrix[10] * dL_dT02;
                float dL_dJ11 = view_matrix[1] * dL_dT10 + view_matrix[5] * dL_dT11 + view_matrix[ 9] * dL_dT12;
                float dL_dJ12 = view_matrix[2] * dL_dT10 + view_matrix[6] * dL_dT11 + view_matrix[10] * dL_dT12;

                // Gradients of loss w.r.t. transformed Gaussian mean t
                float dL_dtx = cache_ptr->x_grad_mul * cache_ptr->t[2] * dL_dJ02;
                float dL_dty = cache_ptr->y_grad_mul * cache_ptr->t[3] * dL_dJ12;
                float dL_dtz = cache_ptr->t[2] * dL_dJ00 + cache_ptr->t[3] * dL_dJ11 + cache_ptr->t[0] * dL_dJ02 + cache_ptr->t[1] * dL_dJ12;

                // Account for transformation of mean to t
                // t = transformPoint4x3(mean, view_matrix);
                float3 dL_dmean = transformVec4x3Transpose(float3{ dL_dtx, dL_dty, dL_dtz }, view_matrix);

                // Gradients of loss w.r.t. Gaussian means, but only the portion
                // that is caused because the mean affects the covariance matrix.
                // Additional mean gradient is accumulated in GSGN::preprocess.
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 0, data);
                out_vec[pos_out] += dL_dmean.x;

                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 1, data);
                out_vec[pos_out] += dL_dmean.y;

                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 2, data);
                out_vec[pos_out] += dL_dmean.z;
            }

            if constexpr(write_cache) {
                // collaboratively write out from cache
                __syncthreads();
                GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache) + blockIdx.x * blockDim.x;
                #pragma unroll
                for(int i = 0; i < NUM_ATTRS; i++) {
                    const uint32_t cache_idx = threadIdx.x + i * 256;
                    const uint32_t tid = cache_idx / NUM_ATTRS;
                    const uint32_t attr_id = cache_idx % NUM_ATTRS;
                    if(tid + blockIdx.x * blockDim.x >= n_visible_gaussians) break;
                    float* linearized_cache_ptr = reinterpret_cast<float*>(cache_ptr + tid);
                    linearized_cache_ptr[attr_id] = cache[tid][attr_id];
                }
            }
        }

        // Backward pass for conversion of spherical harmonics to RGB for
        // each Gaussian.
        template <typename scalar_t, bool write_cache>
        __device__ void computeColorFromSH(
            PackedGSGNDataSpec& data,
            const int dL_pos,
            const int dL_offset,
            const int global_id,
            int deg,
            int max_coeffs,
            const glm::vec3 pos,
            glm::vec3 campos,
            const float* shs,
            const bool* clamped,
            const scalar_t* dL_dcolor,
            scalar_t* out_vec,
            GaussianCachePreprocess* cache_ptr) {

            // TODO: make deg a template parameter again
            
            // Compute intermediate values, as it is done during forward
            glm::vec3 dir_orig = pos - campos;
            glm::vec3 dir = dir_orig / glm::length(dir_orig);

            // sh only starts from second coeff, because sh[0] is not needed in backward passes
            glm::vec3* sh = ((glm::vec3*)shs) + global_id * (max_coeffs - 1);

            // Use PyTorch rule for clamping: if clamping was applied,
            // gradient becomes 0.
            float dL_dRGB[3];
            #pragma unroll
            for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                dL_dRGB[ch] = (float) dL_dcolor[dL_pos + ch * dL_offset];
                dL_dRGB[ch] *= clamped[ch] ? 0 : 1;
            }

            glm::vec3 dRGBdx(0, 0, 0);
            glm::vec3 dRGBdy(0, 0, 0);
            glm::vec3 dRGBdz(0, 0, 0);
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;

            // No tricks here, just high school-level calculus.           
            #pragma unroll
            for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_DC>(global_id, ch, 0, data);
                out_vec[pos_out] += SH_C0 * dL_dRGB[ch];
            }
            if (deg > 0) {
                float dRGBdsh_d1[3];
                dRGBdsh_d1[0] = -SH_C1 * y;
                dRGBdsh_d1[1] = SH_C1 * z;
                dRGBdsh_d1[2] = -SH_C1 * x;
                                    
                #pragma unroll
                for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                    #pragma unroll
                    for(int i = 0; i < 3; i++) {
                        int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);
                        out_vec[pos_out] += dRGBdsh_d1[i] * dL_dRGB[ch];
                    }
                }

                if constexpr(write_cache) {
                    dRGBdx = -SH_C1 * sh[2];
                    dRGBdy = -SH_C1 * sh[0];
                    dRGBdz = SH_C1 * sh[1];
                }

                if (deg > 1) {
                    float xx = x * x, yy = y * y, zz = z * z;
                    float xy = x * y, yz = y * z, xz = x * z;

                    float dRGBdsh_d2[5];
                    dRGBdsh_d2[0] = SH_C2[0] * xy;
                    dRGBdsh_d2[1] = SH_C2[1] * yz;
                    dRGBdsh_d2[2] = SH_C2[2] * (2.f * zz - xx - yy);
                    dRGBdsh_d2[3] = SH_C2[3] * xz;
                    dRGBdsh_d2[4] = SH_C2[4] * (xx - yy);

                    #pragma unroll
                    for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                        #pragma unroll
                        for(int i=3; i < 8; i++) {
                            int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);
                            out_vec[pos_out] += dRGBdsh_d2[i - 3] * dL_dRGB[ch];
                        }
                    }

                    if constexpr(write_cache) {
                        dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
                        dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
                        dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];
                    }

                    if (deg > 2) {
                        float dRGBdsh_d3[7];
                        dRGBdsh_d3[0] = SH_C3[0] * y * (3.f * xx - yy);
                        dRGBdsh_d3[1] = SH_C3[1] * xy * z;
                        dRGBdsh_d3[2] = SH_C3[2] * y * (4.f * zz - xx - yy);
                        dRGBdsh_d3[3] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                        dRGBdsh_d3[4] = SH_C3[4] * x * (4.f * zz - xx - yy);
                        dRGBdsh_d3[5] = SH_C3[5] * z * (xx - yy);
                        dRGBdsh_d3[6] = SH_C3[6] * x * (xx - 3.f * yy);

                        #pragma unroll
                        for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                            #pragma unroll
                            for(int i=8; i < 15; i++) {
                                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);
                                out_vec[pos_out] += dRGBdsh_d3[i - 8] * dL_dRGB[ch];
                            }
                        }

                        if constexpr(write_cache) {
                            dRGBdx += (
                                SH_C3[0] * sh[8] * 3.f * 2.f * xy +
                                SH_C3[1] * sh[9] * yz +
                                SH_C3[2] * sh[10] * -2.f * xy +
                                SH_C3[3] * sh[11] * -3.f * 2.f * xz +
                                SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
                                SH_C3[5] * sh[13] * 2.f * xz +
                                SH_C3[6] * sh[14] * 3.f * (xx - yy));

                            dRGBdy += (
                                SH_C3[0] * sh[8] * 3.f * (xx - yy) +
                                SH_C3[1] * sh[9] * xz +
                                SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
                                SH_C3[3] * sh[11] * -3.f * 2.f * yz +
                                SH_C3[4] * sh[12] * -2.f * xy +
                                SH_C3[5] * sh[13] * -2.f * yz +
                                SH_C3[6] * sh[14] * -3.f * 2.f * xy);

                            dRGBdz += (
                                SH_C3[1] * sh[9] * xy +
                                SH_C3[2] * sh[10] * 4.f * 2.f * yz +
                                SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
                                SH_C3[4] * sh[12] * 4.f * 2.f * xz +
                                SH_C3[5] * sh[13] * (xx - yy));
                        }
                    }
                }
            }

            if constexpr(write_cache) {
                // write to per_gaussian_cache
                #pragma unroll
                for(int i = 0; i < 3; i++) {
                    cache_ptr->dRGBdx[i] = (i == 0) ? dRGBdx.x : (i == 1) ? dRGBdx.y : dRGBdx.z;
                    cache_ptr->dRGBdy[i] = (i == 0) ? dRGBdy.x : (i == 1) ? dRGBdy.y : dRGBdy.z;
                    cache_ptr->dRGBdz[i] = (i == 0) ? dRGBdz.x : (i == 1) ? dRGBdz.y : dRGBdz.z;
                }
            }

            if constexpr(! write_cache) {
                // instead of re-computing dRGBdx in the kernel, we load it from the cache
                dRGBdx.x = cache_ptr->dRGBdx[0];
                dRGBdx.y = cache_ptr->dRGBdx[1];
                dRGBdx.z = cache_ptr->dRGBdx[2];

                dRGBdy.x = cache_ptr->dRGBdy[0];
                dRGBdy.y = cache_ptr->dRGBdy[1];
                dRGBdy.z = cache_ptr->dRGBdy[2];

                dRGBdz.x = cache_ptr->dRGBdz[0];
                dRGBdz.y = cache_ptr->dRGBdz[1];
                dRGBdz.z = cache_ptr->dRGBdz[2];
            }

            // The view direction is an input to the computation. View direction
            // is influenced by the Gaussian's mean, so SHs gradients
            // must propagate back into 3D position.
            float3 dL_ddir{
                dRGBdx.x * dL_dRGB[0] + dRGBdx.y * dL_dRGB[1] + dRGBdx.z * dL_dRGB[2],
                dRGBdy.x * dL_dRGB[0] + dRGBdy.y * dL_dRGB[1] + dRGBdy.z * dL_dRGB[2],
                dRGBdz.x * dL_dRGB[0] + dRGBdz.y * dL_dRGB[1] + dRGBdz.z * dL_dRGB[2]
            };

            // Account for normalization of direction
            float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, dL_ddir);

            // Gradients of loss w.r.t. Gaussian means, but only the portion
            // that is caused because the mean affects the view-dependent color.
            // Additional mean gradient is accumulated in below methods.
            int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 0, data);
            out_vec[pos_out] += dL_dmean.x;

            pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 1, data);
            out_vec[pos_out] += dL_dmean.y;

            pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 2, data);
            out_vec[pos_out] += dL_dmean.z;
        }

        // Backward pass for the conversion of scale and rotation to a 
        // 3D covariance matrix for each Gaussian.
        template <typename scalar_t, bool write_cache>
        __device__ void computeCov3D(
            PackedGSGNDataSpec& data,
            const int dL_pos,
            const int dL_offset,
            const int global_id,
            const float3 unactivated_scale,
            float mod,
            const float4 unactivated_rot,
            const scalar_t* dL_dcov3Ds,
            scalar_t* out_vec,
            GaussianCachePreprocess* cache_ptr) {

            // Recompute (intermediate) results for the 3D covariance computation.
            float4 rot = normv(unactivated_rot);
            float r = rot.x;
            float x = rot.y;
            float y = rot.z;
            float z = rot.w;

            if constexpr(write_cache) {
                // write to per_gaussian_cache
                float R[9] = {
                    1.f - 2.f * (y * y + z * z),
                    2.f * (x * y - r * z),
                    2.f * (x * z + r * y),
                    2.f * (x * y + r * z),
                    1.f - 2.f * (x * x + z * z),
                    2.f * (y * z - r * x),
                    2.f * (x * z - r * y),
                    2.f * (y * z + r * x),
                    1.f - 2.f * (x * x + y * y)
                };

                #pragma unroll
                for(int i = 0; i < 9; i++) {
                    cache_ptr->R[i] = R[i];
                }
            }

            // change inputs once instead of multiple times below
            float3 scale = {mod * exp(unactivated_scale.x), mod * exp(unactivated_scale.y), mod * exp(unactivated_scale.z)};
            r *= 2.f;
            x *= 2.f;
            y *= 2.f;
            z *= 2.f;

            const float dL_dcov[6] = {
                (float) dL_dcov3Ds[dL_pos],
                (float) dL_dcov3Ds[dL_pos + 1 * dL_offset],
                (float) dL_dcov3Ds[dL_pos + 2 * dL_offset],
                (float) dL_dcov3Ds[dL_pos + 3 * dL_offset],
                (float) dL_dcov3Ds[dL_pos + 4 * dL_offset],
                (float) dL_dcov3Ds[dL_pos + 5 * dL_offset]
            };

            float dL_dMt[9] = {
                scale.x * (2.0f * cache_ptr->R[0] * dL_dcov[0] + cache_ptr->R[3] * dL_dcov[1] + cache_ptr->R[6] * dL_dcov[2]),
                scale.x * (cache_ptr->R[0] * dL_dcov[1] + 2.0f * cache_ptr->R[3] * dL_dcov[3] + cache_ptr->R[6] * dL_dcov[4]),
                scale.x * (cache_ptr->R[0] * dL_dcov[2] + cache_ptr->R[3] * dL_dcov[4] + 2.0f * cache_ptr->R[6] * dL_dcov[5]),
                scale.y * (2.0f * cache_ptr->R[1] * dL_dcov[0] + cache_ptr->R[4] * dL_dcov[1] + cache_ptr->R[7] * dL_dcov[2]),
                scale.y * (cache_ptr->R[1] * dL_dcov[1] + 2.0f * cache_ptr->R[4] * dL_dcov[3] + cache_ptr->R[7] * dL_dcov[4]),
                scale.y * (cache_ptr->R[1] * dL_dcov[2] + cache_ptr->R[4] * dL_dcov[4] + 2.0f * cache_ptr->R[7] * dL_dcov[5]),
                scale.z * (2.0f * cache_ptr->R[2] * dL_dcov[0] + cache_ptr->R[5] * dL_dcov[1] + cache_ptr->R[8] * dL_dcov[2]),
                scale.z * (cache_ptr->R[2] * dL_dcov[1] + 2.0f * cache_ptr->R[5] * dL_dcov[3] + cache_ptr->R[8] * dL_dcov[4]),
                scale.z * (cache_ptr->R[2] * dL_dcov[2] + cache_ptr->R[5] * dL_dcov[4] + 2.0f * cache_ptr->R[8] * dL_dcov[5])
            };

            // Gradients of loss w.r.t. scale
            #pragma unroll
            for(int i = 0; i < 3; i++) {
                float dL_ds = cache_ptr->R[i] * dL_dMt[i * 3] + cache_ptr->R[i + 3] * dL_dMt[i * 3 + 1] + cache_ptr->R[i + 6] * dL_dMt[i * 3 + 2];
                float us = (i == 0) ? unactivated_scale.x : (i == 1) ? unactivated_scale.y : unactivated_scale.z;
                dL_ds = dexpvdv(us, dL_ds);
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::SCALE>(global_id, 0, i, data);
                out_vec[pos_out] += dL_ds;
            }

            dL_dMt[0] *= scale.x;
            dL_dMt[1] *= scale.x;
            dL_dMt[2] *= scale.x;
            dL_dMt[3] *= scale.y;
            dL_dMt[4] *= scale.y;
            dL_dMt[5] *= scale.y;
            dL_dMt[6] *= scale.z;
            dL_dMt[7] *= scale.z;
            dL_dMt[8] *= scale.z;

            // Gradients of loss w.r.t. normalized quaternion
            float4 dL_dq;
            dL_dq.x = z * (dL_dMt[1] - dL_dMt[3]) + y * (dL_dMt[6] - dL_dMt[2]) + x * (dL_dMt[5] - dL_dMt[7]);
            dL_dq.y = y * (dL_dMt[3] + dL_dMt[1]) + z * (dL_dMt[6] + dL_dMt[2]) + r * (dL_dMt[5] - dL_dMt[7]) - 2 * x * (dL_dMt[8] + dL_dMt[4]);
            dL_dq.z = x * (dL_dMt[3] + dL_dMt[1]) + r * (dL_dMt[6] - dL_dMt[2]) + z * (dL_dMt[5] + dL_dMt[7]) - 2 * y * (dL_dMt[8] + dL_dMt[0]);
            dL_dq.w = r * (dL_dMt[1] - dL_dMt[3]) + x * (dL_dMt[6] + dL_dMt[2]) + y * (dL_dMt[5] + dL_dMt[7]) - 2 * z * (dL_dMt[4] + dL_dMt[0]);

            // Gradients of loss w.r.t. unnormalized quaternion
            float4 d_rot = dnormvdv(float4{unactivated_rot.x, unactivated_rot.y, unactivated_rot.z, unactivated_rot.w}, dL_dq);

            #pragma unroll
            for(int i=0; i < 4; i++) {
                float grad = (i == 0) ? d_rot.x : (i == 1) ? d_rot.y : (i == 2) ? d_rot.z : d_rot.w;
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::ROTATION>(global_id, 0, i, data);
                out_vec[pos_out] += grad;
            }
        }

        // Backward pass of the preprocessing steps, except
        // for the covariance computation and inversion
        // (those are handled by a previous kernel call)
        template<typename scalar_t, bool write_cache>
        __global__ void __launch_bounds__(256) 
        gsgn_preprocessCUDA(
            PackedGSGNDataSpec data,
            const int img_id,
            const int n_visible_gaussians,
            const scalar_t* dL_dmean2D,
            const scalar_t* dL_dcolor,
            const scalar_t* dL_dcov3D,
            scalar_t* out_vec,
            float* per_gaussian_cache,
            int* map_cache_to_gaussians) {

            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

            constexpr uint32_t ATTR_OFFSET = sizeof(GaussianCacheComputeCov2D) / sizeof(float);
            constexpr uint32_t NUM_ATTRS = sizeof(GaussianCachePreprocess) / sizeof(float);
            __shared__ float cache[256][NUM_ATTRS];
            
            if constexpr(! write_cache) {
                // collaboratively read in dRGBdxyz and R from cache
                GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache) + blockIdx.x * blockDim.x;
                #pragma unroll
                for(int i = 0; i < NUM_ATTRS; i++) {
                    const uint32_t cache_idx = threadIdx.x + i * 256;
                    const uint32_t tid = cache_idx / NUM_ATTRS;
                    const uint32_t attr_id = cache_idx % NUM_ATTRS;
                    if(tid + blockIdx.x * blockDim.x >= n_visible_gaussians) break;
                    float* linearized_cache_ptr = reinterpret_cast<float*>(cache_ptr + tid) + ATTR_OFFSET;
                    cache[tid][attr_id] = linearized_cache_ptr[attr_id];
                }
                __syncthreads();
            }

            if (idx < n_visible_gaussians) {
                const int global_id = map_cache_to_gaussians[idx];
                GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + idx;
                // Don't have to check because we only parallelize over n_visible_gaussians -- all of those have a radius_gt_zero
                // if(! rendered_data_ptr->radius_gt_zero) {
                //     return;
                // }

                GaussianCachePreprocess* cache_ptr = reinterpret_cast<GaussianCachePreprocess*>(cache[threadIdx.x]);

                float3 m = data.means3D[global_id];
                const float* proj = data.projmatrix[img_id]; // todo could live in shared memory

                // Taking care of gradients from the screenspace points
                float4 m_hom = transformPoint4x4(m, proj);
                float m_w = 1.0f / (m_hom.w + 0.0000001f);

                // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
                // from rendering procedure
                glm::vec3 dL_dmean;
                float2 dL_dmean2Ds = {(float) dL_dmean2D[idx], (float) dL_dmean2D[idx + n_visible_gaussians]};
                float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
                float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
                dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2Ds.x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2Ds.y;
                dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2Ds.x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2Ds.y;
                dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2Ds.x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2Ds.y;

                // That's the second part of the mean gradient. Previous computation
                // of cov2D and following SH conversion also affects it.
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 0, data);
                out_vec[pos_out] += dL_dmean.x;

                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 1, data);
                out_vec[pos_out] += dL_dmean.y;

                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, 2, data);
                out_vec[pos_out] += dL_dmean.z;

                // Compute gradient updates due to computing colors from SHs
                glm::vec3 campos = data.campos[img_id]; // todo could live in shared memory
                glm::vec3 mean_vec(m.x, m.y, m.z); // todo could save this copy
                computeColorFromSH<scalar_t, write_cache>(
                    data, idx, n_visible_gaussians, global_id, data.D, data.M, mean_vec, campos, data.shs, rendered_data_ptr->clamped, dL_dcolor, out_vec, cache_ptr
                );

                // Compute gradient updates due to computing covariance from scale/rotation
                computeCov3D<scalar_t, write_cache>(
                    data, idx, n_visible_gaussians, global_id, data.unactivated_scales[global_id], data.scale_modifier, data.unactivated_rotations[global_id], dL_dcov3D, out_vec, cache_ptr
                );
            }

            if constexpr(write_cache) {
                // collaboratively write out dRGBdxyz and R from cache
                __syncthreads();
                GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache) + blockIdx.x * blockDim.x;
                #pragma unroll
                for(int i = 0; i < NUM_ATTRS; i++) {
                    const uint32_t cache_idx = threadIdx.x + i * 256;
                    const uint32_t tid = cache_idx / NUM_ATTRS;
                    const uint32_t attr_id = cache_idx % NUM_ATTRS;
                    if(tid + blockIdx.x * blockDim.x >= n_visible_gaussians) break;
                    float* linearized_cache_ptr = reinterpret_cast<float*>(cache_ptr + tid) + ATTR_OFFSET;
                    linearized_cache_ptr[attr_id] = cache[tid][attr_id];
                }
            }
        }

        template <typename scalar_t, int NUM_SH_COEFF>
        __global__ void __launch_bounds__(128)
        apply_j_resort_x_vec_kernel(
            PackedGSGNDataSpec data,
            scalar_t* __restrict__ x_vec_in,
            scalar_t* __restrict__ x_vec_out) {

            constexpr uint32_t NUM_ATTRS_PER_GAUSSIAN = 3 + 3 + 4 + 1 + 3 * NUM_SH_COEFF;

            const uint32_t global_id = threadIdx.x + blockIdx.x * blockDim.x;

            __shared__ float cache[128][NUM_ATTRS_PER_GAUSSIAN];

            // read coalesced in shmem
            if(global_id < data.P) {
                #pragma unroll
                for(int i = 0; i < NUM_ATTRS_PER_GAUSSIAN; i++) {

                    uint32_t read_idx;

                    if(i < 3) {
                        uint32_t attr_id = i;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, 0, attr_id, data);
                    } else if(i < 6) {
                        uint32_t attr_id = i - 3;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::SCALE>(global_id, 0, attr_id, data);
                    } else if(i < 10) {
                        uint32_t attr_id = i - 6;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::ROTATION>(global_id, 0, attr_id, data);
                    } else if(i < 11) {
                        uint32_t attr_id = i - 10;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::OPACITY>(global_id, 0, attr_id, data);
                    } else if(i < 14) {
                        uint32_t attr_id = i - 11;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_DC>(global_id, attr_id, 0, data);
                    } else {
                        // ch: 0-3, i: 0-14
                        uint32_t attr_id = i - 14;
                        uint32_t sh_idx = attr_id / GSGN_NUM_CHANNELS;
                        uint32_t ch = attr_id % GSGN_NUM_CHANNELS;
                        read_idx = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, sh_idx, data);
                    }

                    // x_vec_out[idx] = x_vec_in[read_idx];
                    cache[threadIdx.x][i] = (float) x_vec_in[read_idx];
                }
            }

            // sync
            __syncthreads();

            // write out coalesced from shmem
            #pragma unroll
            for(int i = 0; i < NUM_ATTRS_PER_GAUSSIAN; i++) {
                const uint32_t curr_idx = i * 128 + threadIdx.x;
                const uint32_t tid = curr_idx / NUM_ATTRS_PER_GAUSSIAN;
                if((blockIdx.x * blockDim.x + tid) >= data.P) break;
                const uint32_t attr_id = curr_idx % NUM_ATTRS_PER_GAUSSIAN;
                x_vec_out[(blockIdx.x * blockDim.x + tid) * NUM_ATTRS_PER_GAUSSIAN + attr_id] = (scalar_t) cache[tid][attr_id];
            }
        }

        template <uint32_t C, typename scalar_t, int NUM_SH_COEFFS>
        __global__ void __launch_bounds__(SPARSE_J_NUM_THREADS)
        apply_j_kernel(
            PackedGSGNDataSpec data,
            scalar_t* __restrict__ x_vec,
            __half** __restrict__ sparse_jacobians,
            int** __restrict__ index_map,
            float** __restrict__ per_gaussian_cache,
            int** __restrict__ segments,
            int** __restrict__ segments_to_gaussian,
            int** __restrict__ num_gaussians_in_block,
            int** __restrict__ block_offset_in_segments,
            int max_gaussians_per_block,
            scalar_t* __restrict__ jx_vec) {

            // this kernel computes J * x by using the intermediate data that was previously cached
            // we store per-image data and per-gaussian data in shared memory to reduce the registers/thread
            // each thread handles one gaussian per ray and has to re-do the calculations to go from intermediate 2D gradients to all gaussian attributes
            // we write out the values by atomically adding to the output position
            // this kernel parallelizes over images in the y-dimension and over num_gaussians in the x-dimension, where num_gaussians is the sum of how many gaussians are volume-rendered per pixel
            // num_gaussians can be a different number per image, so we might waste some threads

            const uint32_t img_id = threadIdx.y + blockIdx.y * blockDim.y;
            if(img_id >= data.num_images) {
                return;
            }
            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t lane_id = threadIdx.x % 32;
            const uint32_t warp_id = threadIdx.x / 32;
            const int stride = data.n_sparse_gaussians[img_id];
            constexpr uint32_t NUM_WARPS = SPARSE_J_NUM_THREADS / 32;
            
            // load per-image attributes into shared memory -- they are used by all threads in this block
            // first warp reads the camera matrices in parallel, all other warsp read a single value with its first thread
            // hopefully, this will make all these reads in parallel
            __shared__ float viewmatrix[16];
            __shared__ float projmatrix[16];
            __shared__ glm::vec3 campos;
            if(threadIdx.x < 16) {
                viewmatrix[threadIdx.x] = data.viewmatrix[img_id][threadIdx.x];
            }
            if(threadIdx.x >= 16 && threadIdx.x < 32) {
                projmatrix[threadIdx.x - 16] = data.projmatrix[img_id][threadIdx.x - 16];
            }
            if(warp_id == 1 || (NUM_WARPS <= 1 && warp_id == 0)) {
                campos = data.campos[img_id];
            }

            // load per-gaussian attributes into shared memory -- they are used by consecutive threads in this block
            // we load them in parallel, which is hopefully faster than if every thread loads it from global memory separately
            extern __shared__ char cache[];
            GeometryStateReduced* rendered_cache = reinterpret_cast<GeometryStateReduced*>(cache);
            GaussianAttributeNoSH* attr_cache = reinterpret_cast<GaussianAttributeNoSH*>(rendered_cache + max_gaussians_per_block);
            GaussianCache* gaussian_cache = reinterpret_cast<GaussianCache*>(attr_cache + max_gaussians_per_block);
            scalar_t* x_vec_cache = reinterpret_cast<scalar_t*>(gaussian_cache + max_gaussians_per_block);
            constexpr uint8_t num_threads_x_vec = 3 + 3 + 4 + 1 + 3 * NUM_SH_COEFFS;
            constexpr uint8_t num_rounds_x_vec = (num_threads_x_vec + 31) / 32;
            const int num_blocks = (stride + SPARSE_J_NUM_THREADS - 1) / SPARSE_J_NUM_THREADS;
            int num_gaussians;
            int segment_idx_block;
            if(blockIdx.x < num_blocks) {
                num_gaussians = num_gaussians_in_block[img_id][blockIdx.x];
                segment_idx_block = block_offset_in_segments[img_id][blockIdx.x];
                constexpr uint8_t num_threads_geom = sizeof(GeometryStateReduced) / 2; // each thread reads 16 Byte
                constexpr uint8_t num_rounds_geom = (num_threads_geom + 31) / 32; // one warp can read one struct completely

                constexpr uint8_t num_threads_attr = sizeof(GaussianAttributeNoSH) / 4; // each thread reads 32 Byte
                constexpr uint8_t num_rounds_attr = (num_threads_attr + 31) / 32; // one warp can read one struct in this many rounds --> sh=1: 1, sh=4: 1, sh=9: 2, sh=16: 2

                constexpr uint8_t num_threads_cache = sizeof(GaussianCache) / 4; // each thread reads 32 Byte
                constexpr uint8_t num_rounds_cache = (num_threads_cache + 31) / 32; // one warp can read one struct completely

                // this if-statement evaluates the same for all warps in the block --> no divergence
                if(num_gaussians * 4 <= NUM_WARPS) {
                    // first warp reads geom, second warp reads attr, third warp reads cache
                    const int gaussian_idx = warp_id / 4;
                    const int read_idx = warp_id % 4;
                    
                    // this if-statement might make some warps do nothing
                    if(gaussian_idx < num_gaussians) {
                        const int gaussian_global_id = segments_to_gaussian[img_id][segment_idx_block + gaussian_idx];

                        // this if-statement evaluates the same for all threads in the warp --> no divergence
                        if(read_idx == 0) {
                            // load next rendered data into shared memory, collaboratively for this warp
                            GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_geom; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_geom) break;
                                reinterpret_cast<uint16_t*>(rendered_cache + gaussian_idx)[read_idx] = rendered_read_in_ptr[read_idx];
                            }
                        } else if(read_idx == 1) {
                            // load next gaussian attributes into shared memory, collaboratively for this warp
                            GaussianAttributeNoSH* attr_ptr = reinterpret_cast<GaussianAttributeNoSH*>(data.params) + gaussian_global_id;
                            float* attr_read_in_ptr = reinterpret_cast<float*>(attr_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_attr; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_attr) break;
                                reinterpret_cast<float*>(attr_cache + gaussian_idx)[read_idx] = attr_read_in_ptr[read_idx];
                            }
                        } else if(read_idx == 2) {
                            // load next gaussian cache into shared memory, collaboratively for this warp
                            GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            float* cache_read_in_ptr = reinterpret_cast<float*>(cache_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_cache; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_cache) break;
                                reinterpret_cast<float*>(gaussian_cache + gaussian_idx)[read_idx] = cache_read_in_ptr[read_idx];
                            }
                        } else if(read_idx == 3) {
                            #pragma unroll
                            for(int k = 0; k < num_rounds_x_vec; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_x_vec) break;
                                x_vec_cache[gaussian_idx * num_threads_x_vec + read_idx] = x_vec[gaussian_global_id * num_threads_x_vec + read_idx];
                            }                         
                        }
                    }
                } else {
                    // all warps first read geom, then read attr, then read cache
                    uint16_t num_rounds_gaussians = (num_gaussians + NUM_WARPS - 1) / NUM_WARPS;

                    #pragma unroll
                    for(uint16_t i = 0; i < num_rounds_gaussians; i++) {
                        // the loop has the same length for all warps in the block --> no divergence
                        const int gaussian_idx = warp_id + i * NUM_WARPS;

                        // this if-statement might make some warps do nothing
                        if(gaussian_idx < num_gaussians) {
                            const int gaussian_global_id = segments_to_gaussian[img_id][segment_idx_block + gaussian_idx];

                            // load next rendered data into shared memory, collaboratively for this warp
                            GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_geom; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_geom) break;
                                reinterpret_cast<uint16_t*>(rendered_cache + gaussian_idx)[read_idx] = rendered_read_in_ptr[read_idx];
                            }

                            // load next gaussian attributes into shared memory, collaboratively for this warp
                            GaussianAttributeNoSH* attr_ptr = reinterpret_cast<GaussianAttributeNoSH*>(data.params) + gaussian_global_id;
                            float* attr_read_in_ptr = reinterpret_cast<float*>(attr_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_attr; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_attr) break;
                                reinterpret_cast<float*>(attr_cache + gaussian_idx)[read_idx] = attr_read_in_ptr[read_idx];
                            }

                            // load next gaussian cache into shared memory, collaboratively for this warp
                            GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            float* cache_read_in_ptr = reinterpret_cast<float*>(cache_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_cache; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_cache) break;
                                reinterpret_cast<float*>(gaussian_cache + gaussian_idx)[read_idx] = cache_read_in_ptr[read_idx];
                            }

                            // load next x_vec into shared memory, collaboratively for this warp
                            #pragma unroll
                            for(int k = 0; k < num_rounds_x_vec; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_x_vec) break;
                                x_vec_cache[gaussian_idx * num_threads_x_vec + read_idx] = x_vec[gaussian_global_id * num_threads_x_vec + read_idx];
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // have at most this many entries in index_map, terminate (can be the case, because we launch a grid with max_n_sparse_gaussians over all images)
            if(idx >= stride) {
                return;
            }

            // find which shared memory location to use for this thread
            // the loop will have only a few iterations, since num_gaussians_in_block should be a small number.
            // similarly, the if statement in the loop will only cause small divergence
            int gaussian_idx = 0;
            for(int i = 1; i < num_gaussians; i++) {
                int s = segments[img_id][segment_idx_block + i];
                if(idx < s) break;
                gaussian_idx++;
            }

            // set pointers to caches at the correct gaussian_idx
            GeometryStateReduced* rendered_data_ptr = rendered_cache + gaussian_idx;
            GaussianAttributeNoSH* attr_ptr = attr_cache + gaussian_idx;
            GaussianCache* gaussian_cache_ptr = gaussian_cache + gaussian_idx;
            scalar_t* x_vec_pos_cache = x_vec_cache + gaussian_idx * num_threads_x_vec;
            scalar_t* x_vec_scale_cache = x_vec_pos_cache + 3;
            scalar_t* x_vec_rot_cache = x_vec_scale_cache + 3;
            scalar_t* x_vec_opacity_cache = x_vec_rot_cache + 4;
            scalar_t* x_vec_feat_dc_cache = x_vec_opacity_cache + 1;
            scalar_t* x_vec_feat_rest_cache = x_vec_feat_dc_cache + 3;

            // Taking care of gradients from the screenspace points
            float m_w = 1.0f / ((projmatrix[3] * attr_ptr->mean3D[0] + projmatrix[7] * attr_ptr->mean3D[1] + projmatrix[11] * attr_ptr->mean3D[2] + projmatrix[15]) + 0.0000001f);
            float mul1 = (projmatrix[0] * attr_ptr->mean3D[0] + projmatrix[4] * attr_ptr->mean3D[1] + projmatrix[8] * attr_ptr->mean3D[2] + projmatrix[12]) * m_w * m_w;
            float mul2 = (projmatrix[1] * attr_ptr->mean3D[0] + projmatrix[5] * attr_ptr->mean3D[1] + projmatrix[9] * attr_ptr->mean3D[2] + projmatrix[13]) * m_w * m_w;

            // compute dRGBdsh once and store in shared mem
            glm::vec3 dir_orig(attr_ptr->mean3D[0] - campos.x, attr_ptr->mean3D[1] - campos.y, attr_ptr->mean3D[2] - campos.z);
            constexpr uint32_t num_entries_dRGBdsh = (NUM_SH_COEFFS >= 16) ? (7 + 5 + 3) : (NUM_SH_COEFFS >= 9) ? (5 + 3) : (NUM_SH_COEFFS >= 4) ? (3) : 1;
            float dRGBdsh_cache[num_entries_dRGBdsh];

            if constexpr(NUM_SH_COEFFS >= 4) {
                glm::vec3 dir = dir_orig / glm::length(dir_orig);
                float x = dir.x;
                float y = dir.y;
                float z = dir.z;

                dRGBdsh_cache[0] = -SH_C1 * y;
                dRGBdsh_cache[1] = SH_C1 * z;
                dRGBdsh_cache[2] = -SH_C1 * x;

                if constexpr(NUM_SH_COEFFS >= 9) {
                    float xx = x * x, yy = y * y, zz = z * z;
                    float xy = x * y, yz = y * z, xz = x * z;

                    dRGBdsh_cache[3] = SH_C2[0] * xy;
                    dRGBdsh_cache[4] = SH_C2[1] * yz;
                    dRGBdsh_cache[5] = SH_C2[2] * (2.f * zz - xx - yy);
                    dRGBdsh_cache[6] = SH_C2[3] * xz;
                    dRGBdsh_cache[7] = SH_C2[4] * (xx - yy);

                    if constexpr(NUM_SH_COEFFS >= 16) {
                        dRGBdsh_cache[8] = SH_C3[0] * y * (3.f * xx - yy);
                        dRGBdsh_cache[9] = SH_C3[1] * xy * z;
                        dRGBdsh_cache[10] = SH_C3[2] * y * (4.f * zz - xx - yy);
                        dRGBdsh_cache[11] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                        dRGBdsh_cache[12] = SH_C3[4] * x * (4.f * zz - xx - yy);
                        dRGBdsh_cache[13] = SH_C3[5] * z * (xx - yy);
                        dRGBdsh_cache[14] = SH_C3[6] * x * (xx - 3.f * yy);
                    }
                }
            }

            // change inputs once instead of multiple times below in the loop
            float3 scale = {
                data.scale_modifier * exp(attr_ptr->unactivated_scale[0]),
                data.scale_modifier * exp(attr_ptr->unactivated_scale[1]),
                data.scale_modifier * exp(attr_ptr->unactivated_scale[2])
            };
            float4 rot = normv(attr_ptr->unactivated_rotation);
            rot.x *= 2.f;
            rot.y *= 2.f;
            rot.z *= 2.f;
            rot.w *= 2.f;

            // find which gaussian and pixel this thread is handeling
            // const int32_t global_id = global_id_cache[gaussian_idx];
            const int ray_id = index_map[img_id][idx];
            jx_vec += ray_id;
            const int pix_id = ray_id - data.num_pixels * img_id;
            const float2 pixf = { (float) (pix_id % data.W), (float) (pix_id / data.W) };

            // Compute blending values, as before.
            float2 d = { rendered_data_ptr->means2D[0] - pixf.x, rendered_data_ptr->means2D[1] - pixf.y };
            float power = -0.5f * (rendered_data_ptr->conic_opacity[0] * d.x * d.x + rendered_data_ptr->conic_opacity[2] * d.y * d.y) - rendered_data_ptr->conic_opacity[1] * d.x * d.y;
            const float G = __expf(power);

            // load values
            GradientCache grad_cache = reinterpret_cast<GradientCache*>(sparse_jacobians[img_id])[idx]; // load all values in one 256 byte read instruction (8 byte per thread * 32 threads)
            const float dchannel_dcolor = __half2float(grad_cache.dchannel_dcolor);

            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * rendered_data_ptr->conic_opacity[0] - gdy * rendered_data_ptr->conic_opacity[1];
            const float dG_ddely = -gdy * rendered_data_ptr->conic_opacity[2] - gdx * rendered_data_ptr->conic_opacity[1];

            #pragma unroll
            for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                // get the gradient w.r.t. alpha of the Gaussian.
                float dL_dalpha = __half2float(grad_cache.dL_dalpha[ch]);

                // Helpful reusable temporary variables
                const float dL_dG = rendered_data_ptr->conic_opacity[3] * dL_dalpha;

                // get gradients w.r.t. 2D mean position of the Gaussian
                float2 dL_dmean2D;
                dL_dmean2D.x = dL_dG * dG_ddelx * 0.5f * data.W;
                dL_dmean2D.y = dL_dG * dG_ddely * 0.5f * data.H;

                // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
                // from rendering procedure
                // That's the second part of the mean gradient. Previous computation
                // of cov2D and following SH conversion also affects it.
                float3 dL_dmean3D;
                dL_dmean3D.x = (projmatrix[0] * m_w - projmatrix[3] * mul1) * dL_dmean2D.x + (projmatrix[1] * m_w - projmatrix[3] * mul2) * dL_dmean2D.y;
                dL_dmean3D.y = (projmatrix[4] * m_w - projmatrix[7] * mul1) * dL_dmean2D.x + (projmatrix[5] * m_w - projmatrix[7] * mul2) * dL_dmean2D.y;
                dL_dmean3D.z = (projmatrix[8] * m_w - projmatrix[11] * mul1) * dL_dmean2D.x + (projmatrix[9] * m_w - projmatrix[11] * mul2) * dL_dmean2D.y;

                // gradient w.r.t opacity
                scalar_t d_opacity = G * dL_dalpha;

                // get gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
                float3 dL_dconic2D;
                dL_dconic2D.x = -0.5f * gdx * d.x * dL_dG;
                dL_dconic2D.y = -0.5f * gdx * d.y * dL_dG;
                dL_dconic2D.z = -0.5f * gdy * d.y * dL_dG;

                // get gradients w.r.t. opacity of the Gaussian
                // cast the float input to scalar_t so the float/double overload is
                // unambiguous when scalar_t == double (clang is stricter than nvcc).
                d_opacity = (scalar_t) dsigmoidvdv((scalar_t) attr_ptr->unactivated_opacity, d_opacity);

                // gradient w.r.t opacity
                scalar_t jx = d_opacity * x_vec_opacity_cache[0];

                // ------------------------------------------------------------
                // from here: merged the code from backward::computeCov2DCUDA |
                // ------------------------------------------------------------

                if(! rendered_data_ptr->radius_gt_zero) {
                    continue;
                }

                // recompute 2D covariance and relevant
                // intermediate forward results needed in the backward.
                float* T = gaussian_cache_ptr->T;
                float dL_da = 0, dL_db = 0, dL_dc = 0;

                float dL_dcov[6] = { 0 };
                if (gaussian_cache_ptr->denom2inv != 0) {
                    // Gradients of loss w.r.t. entries of 2D covariance matrix,
                    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
                    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
                    float tmp1 = gaussian_cache_ptr->b_ * dL_dconic2D.y;
                    float tmp2 = (gaussian_cache_ptr->denom - gaussian_cache_ptr->a_ * gaussian_cache_ptr->c_);
                    dL_da = gaussian_cache_ptr->denom2inv * (-gaussian_cache_ptr->c_ * gaussian_cache_ptr->c_ * dL_dconic2D.x + 2 * gaussian_cache_ptr->c_ * tmp1 + tmp2 * dL_dconic2D.z);
                    dL_dc = gaussian_cache_ptr->denom2inv * (-gaussian_cache_ptr->a_ * gaussian_cache_ptr->a_ * dL_dconic2D.z + 2 * gaussian_cache_ptr->a_ * tmp1 + tmp2 * dL_dconic2D.x);
                    dL_db = gaussian_cache_ptr->denom2inv * 2 * (gaussian_cache_ptr->b_ * gaussian_cache_ptr->c_ * dL_dconic2D.x - (gaussian_cache_ptr->denom + 2 * gaussian_cache_ptr->b_ * gaussian_cache_ptr->b_) * dL_dconic2D.y + gaussian_cache_ptr->a_ * gaussian_cache_ptr->b_ * dL_dconic2D.z);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (diagonal).
                    // cov2D = transpose(T) * transpose(Vrk) * T;
                    dL_dcov[0] = (T[0] * T[0] * dL_da + T[0] * T[3] * dL_db + T[3] * T[3] * dL_dc);
                    dL_dcov[3] = (T[1] * T[1] * dL_da + T[1] * T[4] * dL_db + T[4] * T[4] * dL_dc);
                    dL_dcov[5] = (T[2] * T[2] * dL_da + T[2] * T[5] * dL_db + T[5] * T[5] * dL_dc);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
                    // Off-diagonal elements appear twice --> double the gradient.
                    // cov2D = transpose(T) * transpose(Vrk) * T;
                    dL_dcov[1] = 2 * T[0] * T[1] * dL_da + (T[0] * T[4] + T[1] * T[3]) * dL_db + 2 * T[3] * T[4] * dL_dc;
                    dL_dcov[2] = 2 * T[0] * T[2] * dL_da + (T[0] * T[5] + T[2] * T[3]) * dL_db + 2 * T[3] * T[5] * dL_dc;
                    dL_dcov[4] = 2 * T[2] * T[1] * dL_da + (T[1] * T[5] + T[2] * T[4]) * dL_db + 2 * T[4] * T[5] * dL_dc;
                }

                // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
                // cov2D = transpose(T) * transpose(Vrk) * T;
                float dL_dT00 = 2 * gaussian_cache_ptr->dL_dT_precomp[0] * dL_da + gaussian_cache_ptr->dL_dT_precomp[1] * dL_db;
                float dL_dT10 = 2 * gaussian_cache_ptr->dL_dT_precomp[1] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[0] * dL_db;
                float dL_dT01 = 2 * gaussian_cache_ptr->dL_dT_precomp[2] * dL_da + gaussian_cache_ptr->dL_dT_precomp[3] * dL_db;
                float dL_dT11 = 2 * gaussian_cache_ptr->dL_dT_precomp[3] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[2] * dL_db;
                float dL_dT02 = 2 * gaussian_cache_ptr->dL_dT_precomp[4] * dL_da + gaussian_cache_ptr->dL_dT_precomp[5] * dL_db;
                float dL_dT12 = 2 * gaussian_cache_ptr->dL_dT_precomp[5] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[4] * dL_db;

                // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
                // T = W * J
                float dL_dJ00 = viewmatrix[0] * dL_dT00 + viewmatrix[4] * dL_dT01 + viewmatrix[ 8] * dL_dT02;
                float dL_dJ02 = viewmatrix[2] * dL_dT00 + viewmatrix[6] * dL_dT01 + viewmatrix[10] * dL_dT02;
                float dL_dJ11 = viewmatrix[1] * dL_dT10 + viewmatrix[5] * dL_dT11 + viewmatrix[ 9] * dL_dT12;
                float dL_dJ12 = viewmatrix[2] * dL_dT10 + viewmatrix[6] * dL_dT11 + viewmatrix[10] * dL_dT12;

                // Gradients of loss w.r.t. transformed Gaussian mean t
                float dL_dtx = gaussian_cache_ptr->x_grad_mul * gaussian_cache_ptr->t[2] * dL_dJ02;
                float dL_dty = gaussian_cache_ptr->y_grad_mul * gaussian_cache_ptr->t[3] * dL_dJ12;
                float dL_dtz = gaussian_cache_ptr->t[2] * dL_dJ00 + gaussian_cache_ptr->t[3] * dL_dJ11 + gaussian_cache_ptr->t[0] * dL_dJ02 + gaussian_cache_ptr->t[1] * dL_dJ12;

                // Gradients of loss w.r.t. Gaussian means, but only the portion
                // that is caused because the mean affects the covariance matrix.
                // Additional mean gradient is accumulated in BACKWARD::preprocess.
                float3 dL_dmean3D_p2 = transformVec4x3Transpose(dL_dtx, dL_dty, dL_dtz, viewmatrix);
                dL_dmean3D.x += dL_dmean3D_p2.x;
                dL_dmean3D.y += dL_dmean3D_p2.y;
                dL_dmean3D.z += dL_dmean3D_p2.z;

                // --------------------------------------------------------------
                // from here: merged the code from backward::computeColorFromSH |
                // --------------------------------------------------------------

                // write features_dc gradient
                scalar_t jx_sh = SH_C0 * x_vec_feat_dc_cache[ch];

                // write out all features_rest gradients
                if constexpr(NUM_SH_COEFFS >= 4) {               
                    #pragma unroll
                    for(int i=0; i < 3; i++) {
                        jx_sh += dRGBdsh_cache[i] * x_vec_feat_rest_cache[i * GSGN_NUM_CHANNELS + ch];
                    }

                    if constexpr(NUM_SH_COEFFS >= 9) {
                        #pragma unroll
                        for(int i=3; i < 8; i++) {
                            jx_sh += dRGBdsh_cache[i] * x_vec_feat_rest_cache[i * GSGN_NUM_CHANNELS + ch];
                        }

                        if constexpr(NUM_SH_COEFFS >= 16) {
                            #pragma unroll
                            for(int i=8; i < 15; i++) {
                                jx_sh += dRGBdsh_cache[i] * x_vec_feat_rest_cache[i * GSGN_NUM_CHANNELS + ch];
                            }
                        } // end SH >= 16
                    } //end SH >= 9
                } //end SH >= 4

                // Use PyTorch rule for clamping: if clamping was applied,
                // gradient becomes 0.
                float dL_dcolors = rendered_data_ptr->clamped[ch] ? 0 : dchannel_dcolor;
                jx += dL_dcolors * jx_sh;

                // The view direction is an input to the computation. View direction
                // is influenced by the Gaussian's mean, so SHs gradients
                // must propagate back into 3D position.
                float3 dL_ddir {
                    gaussian_cache_ptr->dRGBdx[ch] * dL_dcolors,
                    gaussian_cache_ptr->dRGBdy[ch] * dL_dcolors,
                    gaussian_cache_ptr->dRGBdz[ch] * dL_dcolors
                };

                // Account for normalization of direction
                float3 dL_dmean = dnormvdv(dir_orig.x, dir_orig.y, dir_orig.z, dL_ddir.x, dL_ddir.y, dL_ddir.z);
                dL_dmean3D.x += dL_dmean.x;
                dL_dmean3D.y += dL_dmean.y;
                dL_dmean3D.z += dL_dmean.z;

                // Gradients of loss w.r.t. Gaussian means, but only the portion
                // that is caused because the mean affects the view-dependent color.
                // Additional mean gradient is accumulated in below methods.
                #pragma unroll
                for(int i=0; i < 3; i++) {
                    scalar_t grad = (i == 0) ? dL_dmean3D.x : (i == 1) ? dL_dmean3D.y : dL_dmean3D.z;
                    jx += grad * x_vec_pos_cache[i];
                }

                // --------------------------------------------------------
                // from here: merged the code from backward::computeCov3D |
                // --------------------------------------------------------

                // Compute loss gradient w.r.t. matrix M
                // fused calculation to use R, apply scaling, and directly use dL_dcov instead of matrix form
                // dSigma_dM = 2 * M
                
                // Recompute (intermediate) results for the 3D covariance computation.
                float dL_dMt[9] = {
                    scale.x * (2.0f * gaussian_cache_ptr->R[0] * dL_dcov[0] + gaussian_cache_ptr->R[3] * dL_dcov[1] + gaussian_cache_ptr->R[6] * dL_dcov[2]),
                    scale.x * (gaussian_cache_ptr->R[0] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[3] * dL_dcov[3] + gaussian_cache_ptr->R[6] * dL_dcov[4]),
                    scale.x * (gaussian_cache_ptr->R[0] * dL_dcov[2] + gaussian_cache_ptr->R[3] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[6] * dL_dcov[5]),
                    scale.y * (2.0f * gaussian_cache_ptr->R[1] * dL_dcov[0] + gaussian_cache_ptr->R[4] * dL_dcov[1] + gaussian_cache_ptr->R[7] * dL_dcov[2]),
                    scale.y * (gaussian_cache_ptr->R[1] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[4] * dL_dcov[3] + gaussian_cache_ptr->R[7] * dL_dcov[4]),
                    scale.y * (gaussian_cache_ptr->R[1] * dL_dcov[2] + gaussian_cache_ptr->R[4] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[7] * dL_dcov[5]),
                    scale.z * (2.0f * gaussian_cache_ptr->R[2] * dL_dcov[0] + gaussian_cache_ptr->R[5] * dL_dcov[1] + gaussian_cache_ptr->R[8] * dL_dcov[2]),
                    scale.z * (gaussian_cache_ptr->R[2] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[5] * dL_dcov[3] + gaussian_cache_ptr->R[8] * dL_dcov[4]),
                    scale.z * (gaussian_cache_ptr->R[2] * dL_dcov[2] + gaussian_cache_ptr->R[5] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[8] * dL_dcov[5])
                };

                // Gradients of loss w.r.t. scale
                #pragma unroll
                for(int i = 0; i < 3; i++) {
                    scalar_t grad = gaussian_cache_ptr->R[i] * dL_dMt[i * 3] + gaussian_cache_ptr->R[i + 3] * dL_dMt[i * 3 + 1] + gaussian_cache_ptr->R[i + 6] * dL_dMt[i * 3 + 2];
                    grad = dexpvdv((scalar_t) attr_ptr->unactivated_scale[i], grad);
                    jx += grad * x_vec_scale_cache[i];
                }

                dL_dMt[0] *= scale.x;
                dL_dMt[1] *= scale.x;
                dL_dMt[2] *= scale.x;
                dL_dMt[3] *= scale.y;
                dL_dMt[4] *= scale.y;
                dL_dMt[5] *= scale.y;
                dL_dMt[6] *= scale.z;
                dL_dMt[7] *= scale.z;
                dL_dMt[8] *= scale.z;

                // Gradients of loss w.r.t. normalized quaternion
                float4 dL_dq;
                dL_dq.x = rot.w * (dL_dMt[1] - dL_dMt[3]) + rot.z * (dL_dMt[6] - dL_dMt[2]) + rot.y * (dL_dMt[5] - dL_dMt[7]);
                dL_dq.y = rot.z * (dL_dMt[3] + dL_dMt[1]) + rot.w * (dL_dMt[6] + dL_dMt[2]) + rot.x * (dL_dMt[5] - dL_dMt[7]) - 2 * rot.y * (dL_dMt[8] + dL_dMt[4]);
                dL_dq.z = rot.y * (dL_dMt[3] + dL_dMt[1]) + rot.x * (dL_dMt[6] - dL_dMt[2]) + rot.w * (dL_dMt[5] + dL_dMt[7]) - 2 * rot.z * (dL_dMt[8] + dL_dMt[0]);
                dL_dq.w = rot.x * (dL_dMt[1] - dL_dMt[3]) + rot.y * (dL_dMt[6] + dL_dMt[2]) + rot.z * (dL_dMt[5] + dL_dMt[7]) - 2 * rot.w * (dL_dMt[4] + dL_dMt[0]);

                // Gradients of loss w.r.t. unnormalized quaternion
                float4 d_rot = dnormvdv(attr_ptr->unactivated_rotation, dL_dq);
                jx += d_rot.x * x_vec_rot_cache[0] + d_rot.y * x_vec_rot_cache[1] + d_rot.z * x_vec_rot_cache[2] + d_rot.w * x_vec_rot_cache[3];

                // add final result to jx_vec
                atomicAdd(&jx_vec[ch * data.jx_stride], jx);
            }
        }

        template <uint32_t C, typename scalar_t, int NUM_SH_COEFFS, GSGN_MODE M>
        __global__ void __launch_bounds__(SPARSE_J_NUM_THREADS)
        apply_jt_kernel(
            PackedGSGNDataSpec data,
            scalar_t* __restrict__ g_vec,
            __half** __restrict__ sparse_jacobians,
            int** __restrict__ index_map,
            float** __restrict__ per_gaussian_cache,
            int** __restrict__ segments,
            int** __restrict__ segments_to_gaussian,
            int** __restrict__ num_gaussians_in_block,
            int** __restrict__ block_offset_in_segments,
            int max_gaussians_per_block,
            scalar_t* __restrict__ jx_vec) {

            // this kernel computes JT * x by using the intermediate data that was previously cached
            // each thread handles one gaussian per ray and has to re-do the calculations to go from intermediate 2D gradients to all gaussian attributes
            // we write out the values by first doing a segmented warp-reduce and then write once to global memory -- this is possible because the threads are sorted by gaussians
            // this kernel parallelizes over images in the y-dimension and over num_gaussians in the x-dimension, where num_gaussians is the sum of how many gaussians are volume-rendered per pixel
            // num_gaussians can be a different number per image, so we might waste some threads

            const uint32_t img_id = threadIdx.y + blockIdx.y * blockDim.y;
            if(img_id >= data.num_images) {
                return;
            }

            // extract values from thread id
            uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t lane_id = threadIdx.x % 32;
            const uint32_t warp_id = threadIdx.x / 32;
            constexpr uint32_t NUM_WARPS = SPARSE_J_NUM_THREADS / 32;
            const int stride = data.n_sparse_gaussians[img_id];

            // load per-image attributes into shared memory -- they are used by all threads in this block
            // first warp reads the camera matrices in parallel, all other warsp read a single value with its first thread
            // hopefully, this will make all these reads in parallel
            __shared__ float viewmatrix[16];
            __shared__ float projmatrix[16];
            __shared__ glm::vec3 campos;

            if(threadIdx.x < 16) {
                viewmatrix[threadIdx.x] = data.viewmatrix[img_id][threadIdx.x];
            }
            if(threadIdx.x >= 16 && threadIdx.x < 32) {
                projmatrix[threadIdx.x - 16] = data.projmatrix[img_id][threadIdx.x - 16];
            }
            if(warp_id == 1 || (NUM_WARPS <= 1 && warp_id == 0)) {
                campos = data.campos[img_id];
            }

            // determine how many gaussians in this image & if the thread is out of bounds
            // 64-bit on HIP so __ballot_sync over the full wavefront is not truncated.
            GSGN_LANE_MASK mask = __ballot_sync(FULL_MASK, idx < stride);

            // load per-gaussian attributes into shared memory -- they are used by consecutive threads in this block
            // we load them in parallel, which is hopefully faster than if every thread loads it from global memory separately
            extern __shared__ char cache[];
            GeometryStateReduced* rendered_cache = reinterpret_cast<GeometryStateReduced*>(cache);
            GaussianAttributeNoSH* attr_cache = reinterpret_cast<GaussianAttributeNoSH*>(rendered_cache + max_gaussians_per_block);
            GaussianCache* gaussian_cache = reinterpret_cast<GaussianCache*>(attr_cache + max_gaussians_per_block);
            int32_t* global_id_cache = reinterpret_cast<int32_t*>(gaussian_cache + max_gaussians_per_block);
            const int num_blocks = (stride + SPARSE_J_NUM_THREADS - 1) / SPARSE_J_NUM_THREADS;
            int num_gaussians;
            int segment_idx_block;
            if(blockIdx.x < num_blocks) {
                num_gaussians = num_gaussians_in_block[img_id][blockIdx.x];
                segment_idx_block = block_offset_in_segments[img_id][blockIdx.x];
                constexpr uint8_t num_threads_geom = sizeof(GeometryStateReduced) / 2; // each thread reads 16 Byte
                constexpr uint8_t num_rounds_geom = (num_threads_geom + 31) / 32; // one warp can read one struct completely

                constexpr uint8_t num_threads_attr = sizeof(GaussianAttributeNoSH) / 4; // each thread reads 32 Byte
                constexpr uint8_t num_rounds_attr = (num_threads_attr + 31) / 32; // one warp can read one struct completely

                constexpr uint8_t num_threads_cache = sizeof(GaussianCache) / 4; // each thread reads 32 Byte
                constexpr uint8_t num_rounds_cache = (num_threads_cache + 31) / 32; // one warp can read one struct completely

                // this if-statement evaluates the same for all warps in the block --> no divergence
                if(num_gaussians * 3 < NUM_WARPS) {
                    // first half of warps reads geom, second half reads attr
                    const int gaussian_idx = warp_id / 3;
                    const int read_idx = warp_id % 3;
                    
                    // this if-statement might make some warps do nothing
                    if(gaussian_idx < num_gaussians) {
                        const int gaussian_global_id = segments_to_gaussian[img_id][segment_idx_block + gaussian_idx];

                        // this if-statement evaluates the same for all threads in the warp --> no divergence
                        if(read_idx == 0) {
                            // load next rendered data into shared memory, collaboratively for this warp
                            GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_geom; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_geom) break;
                                reinterpret_cast<uint16_t*>(rendered_cache + gaussian_idx)[read_idx] = rendered_read_in_ptr[read_idx];
                            }
                        } else if(read_idx == 1) {
                            // load next gaussian attributes into shared memory, collaboratively for this warp
                            GaussianAttributeNoSH* attr_ptr = reinterpret_cast<GaussianAttributeNoSH*>(data.params) + gaussian_global_id;
                            float* attr_read_in_ptr = reinterpret_cast<float*>(attr_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_attr; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_attr) break;
                                reinterpret_cast<float*>(attr_cache + gaussian_idx)[read_idx] = attr_read_in_ptr[read_idx];
                            }
                        } else if(read_idx == 2) {
                            // load next gaussian cache into shared memory, collaboratively for this warp
                            GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            float* cache_read_in_ptr = reinterpret_cast<float*>(cache_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_cache; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_cache) break;
                                reinterpret_cast<float*>(gaussian_cache + gaussian_idx)[read_idx] = cache_read_in_ptr[read_idx];
                            }
                        }

                        // save global_id
                        global_id_cache[gaussian_idx] = gaussian_global_id;
                    }
                } else {
                    // all warps first read geom, then read attr
                    uint16_t num_rounds_gaussians = (num_gaussians + NUM_WARPS - 1) / NUM_WARPS;

                    #pragma unroll
                    for(uint16_t i = 0; i < num_rounds_gaussians; i++) {
                        // the loop has the same length for all warps in the block --> no divergence
                        const int gaussian_idx = warp_id + i * NUM_WARPS;

                        // this if-statement might make some warps do nothing
                        if(gaussian_idx < num_gaussians) {
                            const int gaussian_global_id = segments_to_gaussian[img_id][segment_idx_block + gaussian_idx];

                            // load next rendered data into shared memory, collaboratively for this warp
                            GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_geom; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_geom) break;
                                reinterpret_cast<uint16_t*>(rendered_cache + gaussian_idx)[read_idx] = rendered_read_in_ptr[read_idx];
                            }

                            // load next gaussian attributes into shared memory, collaboratively for this warp
                            GaussianAttributeNoSH* attr_ptr = reinterpret_cast<GaussianAttributeNoSH*>(data.params) + gaussian_global_id;
                            float* attr_read_in_ptr = reinterpret_cast<float*>(attr_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_attr; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_attr) break;
                                reinterpret_cast<float*>(attr_cache + gaussian_idx)[read_idx] = attr_read_in_ptr[read_idx];
                            }

                            // load next gaussian cache into shared memory, collaboratively for this warp
                            GaussianCache* cache_ptr = reinterpret_cast<GaussianCache*>(per_gaussian_cache[img_id]) + data.map_visible_gaussians[img_id][gaussian_global_id];
                            float* cache_read_in_ptr = reinterpret_cast<float*>(cache_ptr);
                            #pragma unroll
                            for(int k = 0; k < num_rounds_cache; k++) {
                                uint32_t read_idx = lane_id + k * 32;
                                if(read_idx >= num_threads_cache) break;
                                reinterpret_cast<float*>(gaussian_cache + gaussian_idx)[read_idx] = cache_read_in_ptr[read_idx];
                            }

                            // save global_id
                            global_id_cache[gaussian_idx] = gaussian_global_id;
                        }
                    }
                }
            }
            __syncthreads();

            // Specialize WarpReduce for type scalar_t. Pin the logical warp width
            // to 32: hipCUB defaults to warpSize (64 on gfx90a) but the segmented
            // reduce is grouped per 32-lane logical warp (lane_id = tid % 32,
            // temp_storage[NUM_WARPS], head_flag from a width-32 shfl_up). An
            // unpinned 64-wide HeadSegmentedSum would span both logical warps and
            // race on the shared TempStorage slot.
            typedef cub::WarpReduce<scalar_t, GSGN_LOGICAL_WARP_SIZE> WarpReduce;

            // Allocate WarpReduce shared memory for all warps
            __shared__ typename WarpReduce::TempStorage temp_storage[NUM_WARPS];

            // have at most this many entries in index_map, terminate (can be the case, because we launch a grid with max_n_sparse_gaussians over all images)
            if(idx >= stride) {
                return;
            }

            // find which shared memory location to use for this thread
            // the loop will have only a few iterations, since num_gaussians_in_block should be a small number.
            // similarly, the if statement in the loop will only cause small divergence
            int gaussian_idx = 0;
            for(int i = 1; i < num_gaussians; i++) {
                int s = segments[img_id][segment_idx_block + i];
                if(idx < s) break;
                gaussian_idx++;
            }
            GeometryStateReduced* rendered_data_ptr = rendered_cache + gaussian_idx;
            GaussianAttributeNoSH* attr_ptr = attr_cache + gaussian_idx;
            GaussianCache* gaussian_cache_ptr = gaussian_cache + gaussian_idx;
            
            // change inputs once instead of multiple times below in the loop
            float3 scale = {
                data.scale_modifier * exp(attr_ptr->unactivated_scale[0]),
                data.scale_modifier * exp(attr_ptr->unactivated_scale[1]),
                data.scale_modifier * exp(attr_ptr->unactivated_scale[2])
            };
            float4 rot = normv(attr_ptr->unactivated_rotation);
            rot.x *= 2.f;
            rot.y *= 2.f;
            rot.z *= 2.f;
            rot.w *= 2.f;

            // determine if we are at the end of a gaussian (assumes index_map is sorted by global_id and ray_id)
            // is FAST because we use the warp intrinsic
            const int32_t global_id = global_id_cache[gaussian_idx];
            // Width-32: confine the head detection to this thread's 32-lane logical
            // warp. On wave64 lane 32 must read its own group's lane 31, not the
            // sibling group's; lane_id == 0 (== tid % 32) forces the group head.
            const int32_t prev_global_id = __shfl_up_sync(mask, global_id, 1, GSGN_LOGICAL_WARP_SIZE);
            const bool head_flag = (lane_id == 0) || (global_id != prev_global_id);

            const int ray_id = index_map[img_id][idx];
            const int pix_id = ray_id - data.num_pixels * img_id;
            const float2 pixf = { (float) (pix_id % data.W), (float) (pix_id / data.W) };

            // load values
            GradientCache grad_cache = reinterpret_cast<GradientCache*>(sparse_jacobians[img_id])[idx]; // load all values in one 256 byte read instruction (8 byte per thread * 32 threads)
            const float dchannel_dcolor = __half2float(grad_cache.dchannel_dcolor);

            #pragma unroll
            for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                // Compute blending values, as before.
                float2 d = { rendered_data_ptr->means2D[0] - pixf.x, rendered_data_ptr->means2D[1] - pixf.y };
                float power = -0.5f * (rendered_data_ptr->conic_opacity[0] * d.x * d.x + rendered_data_ptr->conic_opacity[2] * d.y * d.y) - rendered_data_ptr->conic_opacity[1] * d.x * d.y;
                const float G = __expf(power);

                // get the gradient w.r.t. alpha of the Gaussian.
                float dL_dalpha = __half2float(grad_cache.dL_dalpha[ch]);

                // get the gradient w.r.t. ch-th color channel of the Gaussian.
                float dL_dcolors = dchannel_dcolor;

                scalar_t jx;
                if constexpr(M != GSGN_MODE::PRECONDITIONER) {
                    jx = jx_vec[ch * data.jx_stride + ray_id];
                } else {
                    if(data.have_weights || data.have_weights_ssim) {
                        uint32_t pos = ch * data.jx_stride + img_id * data.num_pixels + pix_id;
                        float s;
                        if(data.have_weights) {
                            // data.weights already is the sum of both weights if weights_ssim is present (we add it before the kernel already). have one less uncoalesced global memory read this way
                            s = data.weights[pos];
                        } else if(data.have_weights_ssim) {
                            s = data.weights_ssim[pos];
                        } else {
                            assert(false);
                        }
                        dL_dalpha *= s;
                        dL_dcolors *= s;
                    }
                }

                // Helpful reusable temporary variables
                const float gdx = G * d.x;
                const float gdy = G * d.y;
                const float dG_ddelx = -gdx * rendered_data_ptr->conic_opacity[0] - gdy * rendered_data_ptr->conic_opacity[1];
                const float dG_ddely = -gdy * rendered_data_ptr->conic_opacity[2] - gdx * rendered_data_ptr->conic_opacity[1];
                const float dL_dG = rendered_data_ptr->conic_opacity[3] * dL_dalpha;

                // get gradients w.r.t. 2D mean position of the Gaussian
                float2 dL_dmean2D;
                dL_dmean2D.x = dL_dG * dG_ddelx * 0.5f * data.W;
                dL_dmean2D.y = dL_dG * dG_ddely * 0.5f * data.H;

                // Taking care of gradients from the screenspace points
                float m_w = 1.0f / ((projmatrix[3] * attr_ptr->mean3D[0] + projmatrix[7] * attr_ptr->mean3D[1] + projmatrix[11] * attr_ptr->mean3D[2] + projmatrix[15]) + 0.0000001f);
                float mul1 = (projmatrix[0] * attr_ptr->mean3D[0] + projmatrix[4] * attr_ptr->mean3D[1] + projmatrix[8] * attr_ptr->mean3D[2] + projmatrix[12]) * m_w * m_w;
                float mul2 = (projmatrix[1] * attr_ptr->mean3D[0] + projmatrix[5] * attr_ptr->mean3D[1] + projmatrix[9] * attr_ptr->mean3D[2] + projmatrix[13]) * m_w * m_w;

                // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
                // from rendering procedure
                // That's the second part of the mean gradient. Previous computation
                // of cov2D and following SH conversion also affects it.
                float3 dL_dmean3D;
                dL_dmean3D.x = (projmatrix[0] * m_w - projmatrix[3] * mul1) * dL_dmean2D.x + (projmatrix[1] * m_w - projmatrix[3] * mul2) * dL_dmean2D.y;
                dL_dmean3D.y = (projmatrix[4] * m_w - projmatrix[7] * mul1) * dL_dmean2D.x + (projmatrix[5] * m_w - projmatrix[7] * mul2) * dL_dmean2D.y;
                dL_dmean3D.z = (projmatrix[8] * m_w - projmatrix[11] * mul1) * dL_dmean2D.x + (projmatrix[9] * m_w - projmatrix[11] * mul2) * dL_dmean2D.y;

                // gradient w.r.t opacity
                scalar_t d_opacity = G * dL_dalpha;

                // get gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
                float3 dL_dconic2D;
                dL_dconic2D.x = -0.5f * gdx * d.x * dL_dG;
                dL_dconic2D.y = -0.5f * gdx * d.y * dL_dG;
                dL_dconic2D.z = -0.5f * gdy * d.y * dL_dG;

                // get gradients w.r.t. opacity of the Gaussian
                // cast the float input to scalar_t so the float/double overload is
                // unambiguous when scalar_t == double (clang is stricter than nvcc).
                d_opacity = (scalar_t) dsigmoidvdv((scalar_t) attr_ptr->unactivated_opacity, d_opacity);

                // opacity grad
                scalar_t grad;
                if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                    grad = d_opacity * d_opacity;
                } else {
                    grad = d_opacity * jx;
                }
                int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::OPACITY>(global_id, ch, 0, data);

                // do warp-reduce over all threads that reference the same gaussian_id
                scalar_t grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                __syncwarp(mask);
                
                // write out value
                // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                if(head_flag) {
                    atomicAdd(&g_vec[pos_out], grad_sum);
                }

                // ------------------------------------------------------------
                // from here: merged the code from backward::computeCov2DCUDA |
                // ------------------------------------------------------------

                if(! rendered_data_ptr->radius_gt_zero) {
                    continue;
                }

                // recompute 2D covariance and relevant
                // intermediate forward results needed in the backward.
                float* T = gaussian_cache_ptr->T;

                float a_ = gaussian_cache_ptr->a_;
                float b_ = gaussian_cache_ptr->b_;
                float c_ = gaussian_cache_ptr->c_;

                float denom = a_ * c_ - b_ * b_;
                float dL_da = 0, dL_db = 0, dL_dc = 0;
                float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

                // Compute intermediate values, as it is done during forward
                glm::vec3 dir_orig(attr_ptr->mean3D[0] - campos.x, attr_ptr->mean3D[1] - campos.y, attr_ptr->mean3D[2] - campos.z);
                glm::vec3 dir = dir_orig / glm::length(dir_orig);

                float dL_dcov[6] = { 0 };

                if(denom2inv != 0) {
                    // Gradients of loss w.r.t. entries of 2D covariance matrix,
                    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
                    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
                    float tmp1 = b_ * dL_dconic2D.y;
                    float tmp2 = (denom - a_ * c_);
                    dL_da = denom2inv * (-c_ * c_ * dL_dconic2D.x + 2 * c_ * tmp1 + tmp2 * dL_dconic2D.z);
                    dL_dc = denom2inv * (-a_ * a_ * dL_dconic2D.z + 2 * a_ * tmp1 + tmp2 * dL_dconic2D.x);
                    dL_db = denom2inv * 2 * (b_ * c_ * dL_dconic2D.x - (denom + 2 * b_ * b_) * dL_dconic2D.y + a_ * b_ * dL_dconic2D.z);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (diagonal).
                    // cov2D = transpose(T) * transpose(Vrk) * T;
                    dL_dcov[0] = (T[0] * T[0] * dL_da + T[0] * T[3] * dL_db + T[3] * T[3] * dL_dc);
                    dL_dcov[3] = (T[1] * T[1] * dL_da + T[1] * T[4] * dL_db + T[4] * T[4] * dL_dc);
                    dL_dcov[5] = (T[2] * T[2] * dL_da + T[2] * T[5] * dL_db + T[5] * T[5] * dL_dc);

                    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
                    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
                    // Off-diagonal elements appear twice --> double the gradient.
                    // cov2D = transpose(T) * transpose(Vrk) * T;
                    dL_dcov[1] = 2 * T[0] * T[1] * dL_da + (T[0] * T[4] + T[1] * T[3]) * dL_db + 2 * T[3] * T[4] * dL_dc;
                    dL_dcov[2] = 2 * T[0] * T[2] * dL_da + (T[0] * T[5] + T[2] * T[3]) * dL_db + 2 * T[3] * T[5] * dL_dc;
                    dL_dcov[4] = 2 * T[2] * T[1] * dL_da + (T[1] * T[5] + T[2] * T[4]) * dL_db + 2 * T[4] * T[5] * dL_dc;
                }

                // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
                // cov2D = transpose(T) * transpose(Vrk) * T;
                float dL_dT00 = 2 * gaussian_cache_ptr->dL_dT_precomp[0] * dL_da + gaussian_cache_ptr->dL_dT_precomp[1] * dL_db;
                float dL_dT10 = 2 * gaussian_cache_ptr->dL_dT_precomp[1] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[0] * dL_db;
                float dL_dT01 = 2 * gaussian_cache_ptr->dL_dT_precomp[2] * dL_da + gaussian_cache_ptr->dL_dT_precomp[3] * dL_db;
                float dL_dT11 = 2 * gaussian_cache_ptr->dL_dT_precomp[3] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[2] * dL_db;
                float dL_dT02 = 2 * gaussian_cache_ptr->dL_dT_precomp[4] * dL_da + gaussian_cache_ptr->dL_dT_precomp[5] * dL_db;
                float dL_dT12 = 2 * gaussian_cache_ptr->dL_dT_precomp[5] * dL_dc + gaussian_cache_ptr->dL_dT_precomp[4] * dL_db;

                // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
                // T = W * J
                float dL_dJ00 = viewmatrix[0] * dL_dT00 + viewmatrix[4] * dL_dT01 + viewmatrix[ 8] * dL_dT02;
                float dL_dJ02 = viewmatrix[2] * dL_dT00 + viewmatrix[6] * dL_dT01 + viewmatrix[10] * dL_dT02;
                float dL_dJ11 = viewmatrix[1] * dL_dT10 + viewmatrix[5] * dL_dT11 + viewmatrix[ 9] * dL_dT12;
                float dL_dJ12 = viewmatrix[2] * dL_dT10 + viewmatrix[6] * dL_dT11 + viewmatrix[10] * dL_dT12;

                // Gradients of loss w.r.t. transformed Gaussian mean t
                float dL_dtx = gaussian_cache_ptr->x_grad_mul * gaussian_cache_ptr->t[2] * dL_dJ02;
                float dL_dty = gaussian_cache_ptr->y_grad_mul * gaussian_cache_ptr->t[3] * dL_dJ12;
                float dL_dtz = gaussian_cache_ptr->t[2] * dL_dJ00 + gaussian_cache_ptr->t[3] * dL_dJ11 + gaussian_cache_ptr->t[0] * dL_dJ02 + gaussian_cache_ptr->t[1] * dL_dJ12;

                // Gradients of loss w.r.t. Gaussian means, but only the portion
                // that is caused because the mean affects the covariance matrix.
                // Additional mean gradient is accumulated in BACKWARD::preprocess.
                float3 dL_dmean3D_p2 = transformVec4x3Transpose(dL_dtx, dL_dty, dL_dtz, viewmatrix);
                dL_dmean3D.x += dL_dmean3D_p2.x;
                dL_dmean3D.y += dL_dmean3D_p2.y;
                dL_dmean3D.z += dL_dmean3D_p2.z;

                // --------------------------------------------------------------
                // from here: merged the code from backward::computeColorFromSH |
                // --------------------------------------------------------------

                // Use PyTorch rule for clamping: if clamping was applied,
                // gradient becomes 0.
                dL_dcolors *= rendered_data_ptr->clamped[ch] ? 0 : 1;

                float x = dir.x;
                float y = dir.y;
                float z = dir.z;

                // write features_dc gradient
                if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                    grad = SH_C0 * dL_dcolors * SH_C0 * dL_dcolors;
                } else {
                    grad = SH_C0 * dL_dcolors * jx;
                }
                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_DC>(global_id, ch, 0, data);

                // do warp-reduce over all threads that reference the same gaussian_id
                grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                // reached after the radius_gt_zero `continue` above: maskless on HIP.
                GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                
                // write out value
                // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                if(head_flag) {
                    atomicAdd(&g_vec[pos_out], grad_sum);
                }

                if constexpr(NUM_SH_COEFFS >= 4) {
                    float dRGBdsh_d1[3];
                    dRGBdsh_d1[0] = -SH_C1 * y;
                    dRGBdsh_d1[1] = SH_C1 * z;
                    dRGBdsh_d1[2] = -SH_C1 * x;
                
                    // write out all sh gradients
                    #pragma unroll
                    for(int i=0; i < 3; i++) {
                        if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                            grad = dRGBdsh_d1[i] * dL_dcolors * dRGBdsh_d1[i] * dL_dcolors;
                        } else {
                            grad = dRGBdsh_d1[i] * dL_dcolors * jx;
                        }
                        pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);

                        // do warp-reduce over all threads that reference the same gaussian_id
                        grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                        // reached after the radius_gt_zero `continue` above: maskless on HIP.
                        GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                        
                        // write out value
                        // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                        if(head_flag) {
                            atomicAdd(&g_vec[pos_out], grad_sum);
                        }
                    }

                    if constexpr(NUM_SH_COEFFS >= 9) {
                        float xx = x * x, yy = y * y, zz = z * z;
                        float xy = x * y, yz = y * z, xz = x * z;

                        float dRGBdsh_d2[5];
                        dRGBdsh_d2[0] = SH_C2[0] * xy;
                        dRGBdsh_d2[1] = SH_C2[1] * yz;
                        dRGBdsh_d2[2] = SH_C2[2] * (2.f * zz - xx - yy);
                        dRGBdsh_d2[3] = SH_C2[3] * xz;
                        dRGBdsh_d2[4] = SH_C2[4] * (xx - yy);

                        // write out all sh gradients
                        #pragma unroll
                        for(int i=3; i < 8; i++) {
                            if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                                grad = dRGBdsh_d2[i - 3] * dL_dcolors * dRGBdsh_d2[i - 3] * dL_dcolors;
                            } else {
                                grad = dRGBdsh_d2[i - 3] * dL_dcolors * jx;
                            }
                            pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);

                            // do warp-reduce over all threads that reference the same gaussian_id
                            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                            // reached after the radius_gt_zero `continue` above: maskless on HIP.
                            GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                            
                            // write out value
                            // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                            if(head_flag) {
                                atomicAdd(&g_vec[pos_out], grad_sum);
                            }
                        }

                        if constexpr(NUM_SH_COEFFS >= 16) {
                            float dRGBdsh_d3[7];
                            dRGBdsh_d3[0] = SH_C3[0] * y * (3.f * xx - yy);
                            dRGBdsh_d3[1] = SH_C3[1] * xy * z;
                            dRGBdsh_d3[2] = SH_C3[2] * y * (4.f * zz - xx - yy);
                            dRGBdsh_d3[3] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                            dRGBdsh_d3[4] = SH_C3[4] * x * (4.f * zz - xx - yy);
                            dRGBdsh_d3[5] = SH_C3[5] * z * (xx - yy);
                            dRGBdsh_d3[6] = SH_C3[6] * x * (xx - 3.f * yy);

                            // write out all sh gradients
                            #pragma unroll
                            for(int i=8; i < 15; i++) {
                                if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                                    grad = dRGBdsh_d3[i - 8] * dL_dcolors * dRGBdsh_d3[i - 8] * dL_dcolors;
                                } else {
                                    grad = dRGBdsh_d3[i - 8] * dL_dcolors * jx;
                                }
                                pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::FEAT_REST>(global_id, ch, i, data);

                                // do warp-reduce over all threads that reference the same gaussian_id
                                grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                                // reached after the radius_gt_zero `continue` above: maskless on HIP.
                                GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                                
                                // write out value
                                // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                                if(head_flag) {
                                    atomicAdd(&g_vec[pos_out], grad_sum);
                                }
                            }
                        } // end SH >= 16
                    } //end SH >= 9
                } //end SH >= 4

                // The view direction is an input to the computation. View direction
                // is influenced by the Gaussian's mean, so SHs gradients
                // must propagate back into 3D position.
                float3 dL_ddir{gaussian_cache_ptr->dRGBdx[ch] * dL_dcolors, gaussian_cache_ptr->dRGBdy[ch] * dL_dcolors, gaussian_cache_ptr->dRGBdz[ch] * dL_dcolors};

                // Account for normalization of direction
                float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, dL_ddir);
                dL_dmean3D.x += dL_dmean.x;
                dL_dmean3D.y += dL_dmean.y;
                dL_dmean3D.z += dL_dmean.z;

                // Gradients of loss w.r.t. Gaussian means, but only the portion
                // that is caused because the mean affects the view-dependent color.
                // Additional mean gradient is accumulated in below methods.
                #pragma unroll
                for(int i=0; i < 3; i++) {
                    grad = (i == 0) ? dL_dmean3D.x : (i == 1) ? dL_dmean3D.y : dL_dmean3D.z;
                    if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                        grad = grad * grad;
                    } else {
                        grad = grad * jx;
                    }
                    pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::POSITION>(global_id, ch, i, data);

                    // do warp-reduce over all threads that reference the same gaussian_id
                    grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                    // reached after the radius_gt_zero `continue` above: maskless on HIP.
                    GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                    
                    // write out value
                    // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                    if(head_flag) {
                        atomicAdd(&g_vec[pos_out], grad_sum);
                    }
                }

                // --------------------------------------------------------------
                // from here: merged the code from backward::computeColorFromSH |
                // --------------------------------------------------------------

                // Compute loss gradient w.r.t. matrix M
                // fused calculation to use R, apply scaling, and directly use dL_dcov instead of matrix form
                // dSigma_dM = 2 * M

                // Recompute (intermediate) results for the 3D covariance computation.
                float dL_dMt[9] = {
                    scale.x * (2.0f * gaussian_cache_ptr->R[0] * dL_dcov[0] + gaussian_cache_ptr->R[3] * dL_dcov[1] + gaussian_cache_ptr->R[6] * dL_dcov[2]),
                    scale.x * (gaussian_cache_ptr->R[0] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[3] * dL_dcov[3] + gaussian_cache_ptr->R[6] * dL_dcov[4]),
                    scale.x * (gaussian_cache_ptr->R[0] * dL_dcov[2] + gaussian_cache_ptr->R[3] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[6] * dL_dcov[5]),
                    scale.y * (2.0f * gaussian_cache_ptr->R[1] * dL_dcov[0] + gaussian_cache_ptr->R[4] * dL_dcov[1] + gaussian_cache_ptr->R[7] * dL_dcov[2]),
                    scale.y * (gaussian_cache_ptr->R[1] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[4] * dL_dcov[3] + gaussian_cache_ptr->R[7] * dL_dcov[4]),
                    scale.y * (gaussian_cache_ptr->R[1] * dL_dcov[2] + gaussian_cache_ptr->R[4] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[7] * dL_dcov[5]),
                    scale.z * (2.0f * gaussian_cache_ptr->R[2] * dL_dcov[0] + gaussian_cache_ptr->R[5] * dL_dcov[1] + gaussian_cache_ptr->R[8] * dL_dcov[2]),
                    scale.z * (gaussian_cache_ptr->R[2] * dL_dcov[1] + 2.0f * gaussian_cache_ptr->R[5] * dL_dcov[3] + gaussian_cache_ptr->R[8] * dL_dcov[4]),
                    scale.z * (gaussian_cache_ptr->R[2] * dL_dcov[2] + gaussian_cache_ptr->R[5] * dL_dcov[4] + 2.0f * gaussian_cache_ptr->R[8] * dL_dcov[5])
                };

                // Gradients of loss w.r.t. scale
                #pragma unroll
                for(int i = 0; i < 3; i++) {
                    grad = gaussian_cache_ptr->R[i] * dL_dMt[i * 3] + gaussian_cache_ptr->R[i + 3] * dL_dMt[i * 3 + 1] + gaussian_cache_ptr->R[i + 6] * dL_dMt[i * 3 + 2];
                    grad = dexpvdv((scalar_t) attr_ptr->unactivated_scale[i], grad);
                    if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                        grad = grad * grad;
                    } else {
                        grad = grad * jx;
                    }
                    pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::SCALE>(global_id, ch, i, data);

                    // do warp-reduce over all threads that reference the same gaussian_id
                    grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                    // reached after the radius_gt_zero `continue` above: maskless on HIP.
                    GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                    
                    // write out value
                    // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                    if(head_flag) {
                        atomicAdd(&g_vec[pos_out], grad_sum);
                    }
                }

                dL_dMt[0] *= scale.x;
                dL_dMt[1] *= scale.x;
                dL_dMt[2] *= scale.x;
                dL_dMt[3] *= scale.y;
                dL_dMt[4] *= scale.y;
                dL_dMt[5] *= scale.y;
                dL_dMt[6] *= scale.z;
                dL_dMt[7] *= scale.z;
                dL_dMt[8] *= scale.z;

                // Gradients of loss w.r.t. normalized quaternion
                float4 dL_dq;
                dL_dq.x = rot.w * (dL_dMt[1] - dL_dMt[3]) + rot.z * (dL_dMt[6] - dL_dMt[2]) + rot.y * (dL_dMt[5] - dL_dMt[7]);
                dL_dq.y = rot.z * (dL_dMt[3] + dL_dMt[1]) + rot.w * (dL_dMt[6] + dL_dMt[2]) + rot.x * (dL_dMt[5] - dL_dMt[7]) - 2 * rot.y * (dL_dMt[8] + dL_dMt[4]);
                dL_dq.z = rot.y * (dL_dMt[3] + dL_dMt[1]) + rot.x * (dL_dMt[6] - dL_dMt[2]) + rot.w * (dL_dMt[5] + dL_dMt[7]) - 2 * rot.z * (dL_dMt[8] + dL_dMt[0]);
                dL_dq.w = rot.x * (dL_dMt[1] - dL_dMt[3]) + rot.y * (dL_dMt[6] + dL_dMt[2]) + rot.z * (dL_dMt[5] + dL_dMt[7]) - 2 * rot.w * (dL_dMt[4] + dL_dMt[0]);

                // Gradients of loss w.r.t. unnormalized quaternion
                float4 d_rot = dnormvdv(attr_ptr->unactivated_rotation, dL_dq);

                #pragma unroll
                for(int i=0; i < 4; i++) {
                    grad = (i == 0) ? d_rot.x : (i == 1) ? d_rot.y : (i == 2) ? d_rot.z : d_rot.w;
                    if constexpr(M == GSGN_MODE::PRECONDITIONER) {
                        grad = grad * grad;
                    } else {
                        grad = grad * jx;
                    }
                    pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::ROTATION>(global_id, ch, i, data);

                    // do warp-reduce over all threads that reference the same gaussian_id
                    grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                    // reached after the radius_gt_zero `continue` above: maskless on HIP.
                    GSGN_SYNCWARP_AFTER_DIVERGENCE(mask);
                    
                    // write out value
                    // is COALESCED because global_id is sequentially increasing and channel/attribute_idx is constant in a block
                    if(head_flag) {
                        atomicAdd(&g_vec[pos_out], grad_sum);
                    }
                }
            }
        }

        template <uint32_t C, typename scalar_t, int NUM_SH_COEFFS>
        __global__ void __launch_bounds__(SPARSE_JT_NUM_THREADS)
        apply_jt_render_bkwd_kernel(
            PackedGSGNDataSpec data,
            scalar_t* __restrict__ g_vec,
            const __half* __restrict__ sparse_jacobians,
            const int* __restrict__ index_map,
            const float* __restrict__ per_gaussian_cache,
            const int* __restrict__ segments,
            const int* __restrict__ segments_to_gaussian,
            const int* __restrict__ num_gaussians_in_block,
            const int* __restrict__ block_offset_in_segments,
            const int max_gaussians_per_block,
            const int stride,
            const int img_id,
            const int dL_offset,
            const scalar_t* __restrict__ jx_vec,
            scalar_t* __restrict__ dL_dcolors,
            scalar_t* __restrict__ dL_dmean2D,
            scalar_t* __restrict__ dL_dconic2D) {

            // TODO: improve shared memory loading, only need mean2D, conic2D. Could move unactivated_opacity outside of kernel (do in python)

            // this kernel computes JT * x by using the intermediate data that was previously cached
            // each thread handles one gaussian per ray and has to re-do the calculations to go from intermediate 2D gradients to all gaussian attributes
            // we write out the values by first doing a segmented warp-reduce and then write once to global memory -- this is possible because the threads are sorted by gaussians
            // this kernel parallelizes over images in the y-dimension and over num_gaussians in the x-dimension, where num_gaussians is the sum of how many gaussians are volume-rendered per pixel
            // num_gaussians can be a different number per image, so we might waste some threads

            // extract values from thread id
            uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t lane_id = threadIdx.x % 32;
            const uint32_t warp_id = threadIdx.x / 32;
            constexpr uint32_t NUM_WARPS = SPARSE_JT_NUM_THREADS / 32;

            // determine how many gaussians in this image & if the thread is out of bounds
            // 64-bit on HIP so __ballot_sync over the full wavefront is not truncated.
            GSGN_LANE_MASK mask = __ballot_sync(FULL_MASK, idx < stride);

            // load per-gaussian attributes into shared memory -- they are used by consecutive threads in this block
            // we load them in parallel, which is hopefully faster than if every thread loads it from global memory separately
            extern __shared__ char cache[];
            GeometryStateReduced* rendered_cache = reinterpret_cast<GeometryStateReduced*>(cache);
            int32_t* global_id_cache = reinterpret_cast<int32_t*>(rendered_cache + max_gaussians_per_block);
            int32_t* dL_pos_cache = reinterpret_cast<int32_t*>(global_id_cache + max_gaussians_per_block);
            float* unactivated_opacity_cache = reinterpret_cast<float*>(dL_pos_cache + max_gaussians_per_block);
            const int num_blocks = (stride + SPARSE_JT_NUM_THREADS - 1) / SPARSE_JT_NUM_THREADS;
            int num_gaussians;
            int segment_idx_block;
            if(blockIdx.x < num_blocks) {
                num_gaussians = num_gaussians_in_block[blockIdx.x];
                segment_idx_block = block_offset_in_segments[blockIdx.x];
                constexpr uint8_t num_threads_geom = sizeof(GeometryStateReduced) / 2; // each thread reads 16 Byte
                constexpr uint8_t num_rounds_geom = (num_threads_geom + 31) / 32; // one warp can read one struct completely

                uint16_t num_rounds_gaussians = (num_gaussians + NUM_WARPS - 1) / NUM_WARPS;

                #pragma unroll
                for(uint16_t i = 0; i < num_rounds_gaussians; i++) {
                    // the loop has the same length for all warps in the block --> no divergence
                    const int gaussian_idx = warp_id + i * NUM_WARPS;

                    // this if-statement might make some warps do nothing
                    if(gaussian_idx < num_gaussians) {
                        const int gaussian_global_id = segments_to_gaussian[segment_idx_block + gaussian_idx];
                        const int dL_pos = data.map_visible_gaussians[img_id][gaussian_global_id];

                        // load next rendered data into shared memory, collaboratively for this warp
                        GeometryStateReduced* rendered_data_ptr = reinterpret_cast<GeometryStateReduced*>(data.geomBuffer_ptrs[img_id]) + dL_pos;
                        uint16_t* rendered_read_in_ptr = reinterpret_cast<uint16_t*>(rendered_data_ptr);
                        #pragma unroll
                        for(int k = 0; k < num_rounds_geom; k++) {
                            uint32_t read_idx = lane_id + k * 32;
                            if(read_idx >= num_threads_geom) break;
                            reinterpret_cast<uint16_t*>(rendered_cache + gaussian_idx)[read_idx] = rendered_read_in_ptr[read_idx];
                        }

                        // save global_id
                        global_id_cache[gaussian_idx] = gaussian_global_id;

                        // save dL_pos
                        dL_pos_cache[gaussian_idx] = dL_pos;

                        // save unactivated opacity
                        unactivated_opacity_cache[gaussian_idx] = data.unactivated_opacity[gaussian_global_id];
                    }
                }
            }
            __syncthreads();

            // Specialize WarpReduce for type scalar_t. Pin the logical warp width
            // to 32: hipCUB defaults to warpSize (64 on gfx90a) but the segmented
            // reduce is grouped per 32-lane logical warp (lane_id = tid % 32,
            // temp_storage[NUM_WARPS], head_flag from a width-32 shfl_up). An
            // unpinned 64-wide HeadSegmentedSum would span both logical warps and
            // race on the shared TempStorage slot.
            typedef cub::WarpReduce<scalar_t, GSGN_LOGICAL_WARP_SIZE> WarpReduce;

            // Allocate WarpReduce shared memory for all warps
            __shared__ typename WarpReduce::TempStorage temp_storage[NUM_WARPS];

            // have at most this many entries in index_map, terminate (can be the case, because we launch a grid with max_n_sparse_gaussians over all images)
            if(idx >= stride) {
                return;
            }

            // find which shared memory location to use for this thread
            // the loop will have only a few iterations, since num_gaussians_in_block should be a small number.
            // similarly, the if statement in the loop will only cause small divergence
            int gaussian_idx = 0;
            for(int i = 1; i < num_gaussians; i++) {
                int s = segments[segment_idx_block + i];
                if(idx < s) break;
                gaussian_idx++;
            }
            GeometryStateReduced* rendered_data_ptr = rendered_cache + gaussian_idx;
            
            // determine if we are at the end of a gaussian (assumes index_map is sorted by global_id and ray_id)
            // is FAST because we use the warp intrinsic
            const int32_t global_id = global_id_cache[gaussian_idx];
            // Width-32: confine the head detection to this thread's 32-lane logical
            // warp. On wave64 lane 32 must read its own group's lane 31, not the
            // sibling group's; lane_id == 0 (== tid % 32) forces the group head.
            const int32_t prev_global_id = __shfl_up_sync(mask, global_id, 1, GSGN_LOGICAL_WARP_SIZE);
            const bool head_flag = (lane_id == 0) || (global_id != prev_global_id);

            const int ray_id = index_map[idx];
            const int pix_id = ray_id - data.num_pixels * img_id;
            const float2 pixf = { (float) (pix_id % data.W), (float) (pix_id / data.W) };

            // load values
            GradientCache grad_cache = reinterpret_cast<const GradientCache*>(sparse_jacobians)[idx]; // load all values in one 256 byte read instruction (8 byte per thread * 32 threads)
            const float dchannel_dcolor = __half2float(grad_cache.dchannel_dcolor);

            // Compute blending values, as before.
            float2 d = { rendered_data_ptr->means2D[0] - pixf.x, rendered_data_ptr->means2D[1] - pixf.y };
            float power = -0.5f * (rendered_data_ptr->conic_opacity[0] * d.x * d.x + rendered_data_ptr->conic_opacity[2] * d.y * d.y) - rendered_data_ptr->conic_opacity[1] * d.x * d.y;
            const float G = __expf(power);

            const int dL_pos = dL_pos_cache[gaussian_idx];

            float dL_dalpha = 0.0f;
            #pragma unroll
            for(int ch = 0; ch < GSGN_NUM_CHANNELS; ch++) {
                scalar_t jx = jx_vec[ch * data.jx_stride + ray_id];

                // get the gradient w.r.t. alpha of the Gaussian.
                dL_dalpha += __half2float(grad_cache.dL_dalpha[ch]) * jx;

                // Update the gradient w.r.t. ch-th color channel of the Gaussian.
                scalar_t grad = dchannel_dcolor * jx;

                // do warp-reduce over all threads that reference the same gaussian_id
                scalar_t grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
                __syncwarp(mask);

                // write out value
                if(head_flag) {
                    atomicAdd(&dL_dcolors[dL_pos + ch * dL_offset], grad_sum);
                }
            }

            // Helpful reusable temporary variables
            const float dL_dG = rendered_data_ptr->conic_opacity[3] * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * rendered_data_ptr->conic_opacity[0] - gdy * rendered_data_ptr->conic_opacity[1];
            const float dG_ddely = -gdy * rendered_data_ptr->conic_opacity[2] - gdx * rendered_data_ptr->conic_opacity[1];

            // Gradient of pixel coordinate w.r.t. normalized
            // screen-space viewport corrdinates (-1 to 1)
            const float ddelx_dx = 0.5f * data.W;
            const float ddely_dy = 0.5f * data.H;
            
            // Update gradients w.r.t. 2D mean position of the Gaussian
            scalar_t grad = dL_dG * dG_ddelx * ddelx_dx;
            scalar_t grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&dL_dmean2D[dL_pos], grad_sum);
            }

            grad = dL_dG * dG_ddely * ddely_dy;
            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&dL_dmean2D[dL_pos + dL_offset], grad_sum);
            }

            // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
            grad = -0.5f * gdx * d.x * dL_dG;
            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&dL_dconic2D[dL_pos], grad_sum);
            }

            grad = -0.5f * gdx * d.y * dL_dG;
            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&dL_dconic2D[dL_pos + dL_offset], grad_sum);
            }

            grad = -0.5f * gdy * d.y * dL_dG;
            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(grad, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&dL_dconic2D[dL_pos + 2 * dL_offset], grad_sum);
            }

            // Update gradient w.r.t opacity
            scalar_t d_opacity = dsigmoidvdv(unactivated_opacity_cache[gaussian_idx], G * dL_dalpha);
            int32_t pos_out = get_vector_position<GAUSSIAN_ATTRIBUTE::OPACITY>(global_id, 0, 0, data);
            grad_sum = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(d_opacity, head_flag);
            __syncwarp(mask);
            if(head_flag) {
                atomicAdd(&g_vec[pos_out], grad_sum);
            }
        }

        template <uint32_t C, typename scalar_t>
        __global__ void __launch_bounds__(256)
        gsgn_sort_sparse_jacobians_kernel(
            PackedGSGNDataSpec data,
            scalar_t** __restrict__ in_sparse_jacobians,
            scalar_t** __restrict__ out_sparse_jacobians,
            int64_t** __restrict__ indices) {

            constexpr uint32_t num_threads_per_entry = sizeof(GradientCache);
            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t img_id = threadIdx.y + blockIdx.y * blockDim.y;

            if(img_id >= data.num_images) {
                return;
            }

            const int stride = data.n_sparse_gaussians[img_id];
            if(idx >= stride * num_threads_per_entry) {
                return;
            }

            const uint32_t lane_id = idx % num_threads_per_entry;
            const uint32_t entry_id = idx / num_threads_per_entry;

            const uint64_t read_idx = indices[img_id][entry_id];

            // set pointer to correct position to read/write at
            GradientCache* in_data_ptr = reinterpret_cast<GradientCache*>(in_sparse_jacobians[img_id]) + read_idx;
            GradientCache* out_data_ptr = reinterpret_cast<GradientCache*>(out_sparse_jacobians[img_id]) + entry_id;

            // every thread reads one value (== 1 Byte) of the entry.
            // subsequent threads read/write subsequent positions in global memory --> coalesced pattern (except at entry borders for read)
            // Even better: since sizeof(GeometryStateReduced) == 64, one warp handles one entry and everything is perfectly coalesced
            reinterpret_cast<char*>(out_data_ptr)[lane_id] = reinterpret_cast<char*>(in_data_ptr)[lane_id];
        }

        __global__ void __launch_bounds__(256)
        gsgn_fill_reordered_geometry_buffer_kernel(
            const int P,
            const bool* __restrict__ clamped,
            const int* __restrict__ radii,
            const float2* __restrict__ means2D,
            const float* __restrict__ cov3Ds,
            const float4* __restrict__ conic_opacity,
            const float* __restrict__ colors,
            char* __restrict__ out_ptr) {

            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= P) {
                return;
            }

            // set pointer to correct position to write at
            GeometryStateReduced* data_ptr = reinterpret_cast<GeometryStateReduced*>(out_ptr);
            data_ptr = data_ptr + idx;

            // write out clamped
            #pragma unroll
            for(int i = 0; i < GSGN_NUM_CHANNELS; i++) {
                data_ptr->clamped[i] = clamped[GSGN_NUM_CHANNELS * idx + i];
            }
            
            // write out radius -- convert to only bool > 0
            data_ptr->radius_gt_zero = radii[idx] > 0;

            // write out means2D
            data_ptr->means2D[0] = means2D[idx].x;
            data_ptr->means2D[1] = means2D[idx].y;

            // write out cov3Ds
            #pragma unroll
            for(int i = 0; i < 6; i++) {
                data_ptr->cov3D[i] = cov3Ds[6 * idx + i];
            }

            // write out conic_opacity
            data_ptr->conic_opacity[0] = conic_opacity[idx].x;
            data_ptr->conic_opacity[1] = conic_opacity[idx].y;
            data_ptr->conic_opacity[2] = conic_opacity[idx].z;
            data_ptr->conic_opacity[3] = conic_opacity[idx].w;

            // write out colors
            #pragma unroll
            for(int i = 0; i < GSGN_NUM_CHANNELS; i++) {
                data_ptr->rgb[i] = colors[GSGN_NUM_CHANNELS * idx + i];
            }
        }

        __global__ void __launch_bounds__(256)
        gsgn_filter_reordered_geometry_buffer_kernel(
            const int num_visible_gaussians,
            const int* __restrict__ map_cache_to_gaussians,
            char* __restrict__ in_ptr,
            char* __restrict__ out_ptr) {

            constexpr uint32_t num_threads_per_entry = sizeof(GeometryStateReduced) / 2;
            const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= num_visible_gaussians * num_threads_per_entry) {
                return;
            }
            const uint32_t lane_id = idx % num_threads_per_entry;
            const uint32_t entry_id = idx / num_threads_per_entry;

            // get input position
            const uint32_t input_idx = map_cache_to_gaussians[entry_id];

            // set pointer to correct position to read/write at
            GeometryStateReduced* in_data_ptr = reinterpret_cast<GeometryStateReduced*>(in_ptr) + input_idx;
            GeometryStateReduced* out_data_ptr = reinterpret_cast<GeometryStateReduced*>(out_ptr) + entry_id;

            // every thread reads one value (== 2 Byte) of the entry.
            // subsequent threads read/write subsequent positions in global memory --> coalesced pattern (except at entry borders for read)
            // Even better: since sizeof(GeometryStateReduced) == 64, one warp handles one entry and everything is perfectly coalesced
            reinterpret_cast<uint16_t*>(out_data_ptr)[lane_id] = reinterpret_cast<uint16_t*>(in_data_ptr)[lane_id];
        }
    }
}

void CudaRasterizer::GSGN::fill_reordered_geometry_buffer(
    const int P,
    const bool* clamped,
    const int* radii,
    const float2* means2D,
    const float* cov3Ds,
    const float4* conic_opacity,
    const float* colors,
    char* out_ptr) {

    gsgn_fill_reordered_geometry_buffer_kernel<<<(P + 255) / 256, 256>>>(
        P, clamped, radii, means2D, cov3Ds, conic_opacity, colors, out_ptr
    );
}

void CudaRasterizer::GSGN::filter_reordered_geometry_buffer(const int num_visible_gaussians, int* map_cache_to_gaussians, char* geom_buffer, char* out_geom_buffer) {
    gsgn_filter_reordered_geometry_buffer_kernel<<<(num_visible_gaussians * (sizeof(GeometryStateReduced) / 2) + 255) / 256, 256>>>(
        num_visible_gaussians, map_cache_to_gaussians, geom_buffer, out_geom_buffer
    );
}

// eval_jtf_and_get_sparse_jacobian
template<typename T> void CudaRasterizer::GSGN::eval_jtf_and_get_sparse_jacobian(PackedGSGNDataSpec& data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache) {
    // alloc additional memory for the intermediate outputs
    T* helper_memory;
    cudaMallocManaged((void**) &helper_memory, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS + 6) * sizeof(T));

    T* dL_dmeans2D = helper_memory;
    T* dL_dconic2D = dL_dmeans2D + data.max_n_visible_gaussians * 2;
    T* dL_dcolors = dL_dconic2D + data.max_n_visible_gaussians * 3;
    T* dL_dcov3D = dL_dcolors + data.max_n_visible_gaussians * GSGN_NUM_CHANNELS;

    dim3 block_rest = dim3(256, 1, 1);
    dim3 block_render = dim3(GSGN_BLOCK_X, GSGN_BLOCK_Y, 1);
    dim3 grid_render = dim3((data.W + block_render.x - 1) / block_render.x, (data.H + block_render.y - 1) / block_render.y, 1);

    for(int i = 0; i < data.num_images; i++) {
        int dL_offset = data.n_visible_gaussians[i];

        // reset to zeros (except dL_dcov3D)
        cudaMemset(helper_memory, 0, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS) * sizeof(T));

        // Propagate gradients to 2D mean, conic and opacity
        eval_jtf_sparse_render_bkwd_kernel<T><<<grid_render, block_render>>>(
            data, i, dL_offset, r_vec, dL_dcolors, dL_dmeans2D, dL_dconic2D, sparse_jacobians[i], index_map[i]
        );

        dim3 grid_rest = dim3((dL_offset + 255) / 256, 1, 1);

        // Propagate gradients for the path of 2D conic matrix computation.
        // Somewhat long, thus it is its own kernel rather than being part of
        // "preprocess". When done, loss gradient w.r.t. 3D means has been
        // modified and gradient w.r.t. 3D covariance matrix has been computed.
        gsgn_computeCov2DCUDA<T, true><<<grid_rest, block_rest>>>(
            data, i, dL_offset, dL_dconic2D, r_vec, dL_dcov3D, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );

        // Propagate gradients for remaining steps: finish 3D mean gradients,
        // propagate color gradients to SH (if desired), propagate 3D covariance
        // matrix gradients to scale and rotation.
        gsgn_preprocessCUDA<T, true><<<grid_rest, block_rest>>>(
            data, i, dL_offset, dL_dmeans2D, dL_dcolors, dL_dcov3D, r_vec, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );
    }

    // free additional memory for the intermediate outputs
    cudaFree(helper_memory);
}
template void CudaRasterizer::GSGN::eval_jtf_and_get_sparse_jacobian<float>(PackedGSGNDataSpec& data, float* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);
template void CudaRasterizer::GSGN::eval_jtf_and_get_sparse_jacobian<double>(PackedGSGNDataSpec& data, double* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);

// apply_j
template<typename T> void CudaRasterizer::GSGN::apply_j(PackedGSGNDataSpec& data, T* x_vec, T* x_resorted_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr) {
    dim3 grid = dim3((data.max_n_sparse_gaussians + SPARSE_J_NUM_THREADS - 1) / SPARSE_J_NUM_THREADS, data.num_images, 1);
    dim3 block = dim3(SPARSE_J_NUM_THREADS, 1, 1);

    if(data.D == 0) {
        assert(data.M == 1);
        constexpr uint32_t num_attrs_per_gaussian = 3 + 3 + 4 + 1 + 3 * 1;
        constexpr uint32_t bytes_x_vec_per_gaussian = num_attrs_per_gaussian * sizeof(T);
        constexpr uint32_t bytes_per_gaussian = sizeof(GeometryStateReduced) + sizeof(GaussianCache) + bytes_x_vec_per_gaussian + sizeof(GaussianAttributeNoSH);
        apply_j_resort_x_vec_kernel<T, 1><<<(data.P + 127) / 128, 128>>>(
            data, x_vec, x_resorted_vec
        );
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_j_kernel<GSGN_NUM_CHANNELS, T, 1><<<grid, block, size_shared_memory>>>(
            data, x_resorted_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, jx_vec
        );
    } else if(data.D == 1) {
        assert(data.M == 4);
        constexpr uint32_t num_attrs_per_gaussian = 3 + 3 + 4 + 1 + 3 * 4;
        constexpr uint32_t bytes_x_vec_per_gaussian = num_attrs_per_gaussian * sizeof(T);
        constexpr uint32_t bytes_per_gaussian = sizeof(GeometryStateReduced) + sizeof(GaussianCache) + bytes_x_vec_per_gaussian + sizeof(GaussianAttributeNoSH);
        apply_j_resort_x_vec_kernel<T, 4><<<(data.P + 127) / 128, 128>>>(
            data, x_vec, x_resorted_vec
        );
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_j_kernel<GSGN_NUM_CHANNELS, T, 4><<<grid, block, size_shared_memory>>>(
            data, x_resorted_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, jx_vec
        );
    } else if(data.D == 2) {
        assert(data.M == 9);
        constexpr uint32_t num_attrs_per_gaussian = 3 + 3 + 4 + 1 + 3 * 9;
        constexpr uint32_t bytes_x_vec_per_gaussian = num_attrs_per_gaussian * sizeof(T);
        constexpr uint32_t bytes_per_gaussian = sizeof(GeometryStateReduced) + sizeof(GaussianCache) + bytes_x_vec_per_gaussian + sizeof(GaussianAttributeNoSH);
        apply_j_resort_x_vec_kernel<T, 9><<<(data.P + 127) / 128, 128>>>(
            data, x_vec, x_resorted_vec
        );
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_j_kernel<GSGN_NUM_CHANNELS, T, 9><<<grid, block, size_shared_memory>>>(
            data, x_resorted_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, jx_vec
        );
    } else if(data.D == 3) {
        assert(data.M == 16);
        constexpr uint32_t num_attrs_per_gaussian = 3 + 3 + 4 + 1 + 3 * 16;
        constexpr uint32_t bytes_x_vec_per_gaussian = num_attrs_per_gaussian * sizeof(T);
        constexpr uint32_t bytes_per_gaussian = sizeof(GeometryStateReduced) + sizeof(GaussianCache) + bytes_x_vec_per_gaussian + sizeof(GaussianAttributeNoSH);
        apply_j_resort_x_vec_kernel<T, 16><<<(data.P + 127) / 128, 128>>>(
            data, x_vec, x_resorted_vec
        );
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_j_kernel<GSGN_NUM_CHANNELS, T, 16><<<grid, block, size_shared_memory>>>(
            data, x_resorted_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, jx_vec
        );
    }
}
template void CudaRasterizer::GSGN::apply_j<float>(PackedGSGNDataSpec& data, float* x_vec, float* x_resorted_vec, float* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);
template void CudaRasterizer::GSGN::apply_j<double>(PackedGSGNDataSpec& data, double* x_vec, double* x_resorted_vec, double* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

// apply_jt
template<typename T> void CudaRasterizer::GSGN::apply_jt(PackedGSGNDataSpec& data, T* g_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr) {
    constexpr uint32_t bytes_needed = (sizeof(GeometryStateReduced) + sizeof(float) + 2 * sizeof(int32_t));

    // alloc additional memory for the intermediate outputs
    // TODO: keep allocated the whole time?
    T* helper_memory;
    cudaMallocManaged((void**) &helper_memory, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS + 6) * sizeof(T));

    T* dL_dmeans2D = helper_memory;
    T* dL_dconic2D = helper_memory + data.max_n_visible_gaussians * 2;
    T* dL_dcolors = helper_memory + data.max_n_visible_gaussians * (2 + 3);
    T* dL_dcov3D = helper_memory + data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS);

    for(int i = 0; i < data.num_images; i++) {
        int dL_offset = data.n_visible_gaussians[i];

        // reset to zeros (except dL_dcov3D)
        cudaMemset(helper_memory, 0, data.max_n_visible_gaussians * (2 + 3 + GSGN_NUM_CHANNELS) * sizeof(T));

        // Propagate gradients to 2D mean, conic and opacity
        int stride = data.n_sparse_gaussians[i];
        dim3 grid = dim3((stride + SPARSE_JT_NUM_THREADS - 1) / SPARSE_JT_NUM_THREADS, 1, 1);
        dim3 block = dim3(SPARSE_JT_NUM_THREADS, 1, 1);
        int max_gaussians_per_block_per_image = max_gaussians_per_block_per_image_ptr[i];
        uint32_t size_shared_memory = max_gaussians_per_block_per_image * bytes_needed;
        apply_jt_render_bkwd_kernel<GSGN_NUM_CHANNELS, T, 1><<<grid, block, size_shared_memory>>>(
            data, g_vec, sparse_jacobians[i], index_map[i], per_gaussian_cache[i], segments[i], segments_to_gaussians[i], num_gaussians_in_block[i], block_offset_in_segments[i], max_gaussians_per_block_per_image, stride, i, dL_offset, jx_vec, dL_dcolors, dL_dmeans2D, dL_dconic2D
        );

        // Propagate gradients for the path of 2D conic matrix computation.
        // Somewhat long, thus it is its own kernel rather than being part of
        // "preprocess". When done, loss gradient w.r.t. 3D means has been
        // modified and gradient w.r.t. 3D covariance matrix has been computed.
        grid = dim3((dL_offset + 255) / 256, 1, 1);
        block = dim3(256, 1, 1);
        gsgn_computeCov2DCUDA<T, false><<<grid, block>>>(
            data, i, dL_offset, dL_dconic2D, g_vec, dL_dcov3D, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );

        // Propagate gradients for remaining steps: finish 3D mean gradients,
        // propagate color gradients to SH (if desired), propagate 3D covariance
        // matrix gradients to scale and rotation.
        gsgn_preprocessCUDA<T, false><<<grid, block>>>(
            data, i, dL_offset, dL_dmeans2D, dL_dcolors, dL_dcov3D, g_vec, per_gaussian_cache[i], data.map_cache_to_gaussians[i]
        );
    }

    // free additional memory for the intermediate outputs
    cudaFree(helper_memory);
}
template void CudaRasterizer::GSGN::apply_jt<float>(PackedGSGNDataSpec& data, float* g_vec, float* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);
template void CudaRasterizer::GSGN::apply_jt<double>(PackedGSGNDataSpec& data, double* g_vec, double* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

template<typename T> void CudaRasterizer::GSGN::calc_preconditioner(PackedGSGNDataSpec& data, T* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block) {
    dim3 grid = dim3((data.max_n_sparse_gaussians + SPARSE_JT_NUM_THREADS - 1) / SPARSE_JT_NUM_THREADS, data.num_images, 1);
    dim3 block = dim3(SPARSE_JT_NUM_THREADS, 1, 1);
    constexpr uint32_t bytes_needed = (sizeof(GeometryStateReduced) + sizeof(GaussianCache) + sizeof(int32_t));
    if(data.D == 0) {
        assert(data.M == 1);
        constexpr uint32_t bytes_per_gaussian = bytes_needed + sizeof(GaussianAttributeNoSH);
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_jt_kernel<GSGN_NUM_CHANNELS, T, 1, GSGN_MODE::PRECONDITIONER><<<grid, block, size_shared_memory>>>(
            data, M_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, nullptr
        );
    } else if(data.D == 1) {
        assert(data.M == 4);
        constexpr uint32_t bytes_per_gaussian = bytes_needed + sizeof(GaussianAttributeNoSH);
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_jt_kernel<GSGN_NUM_CHANNELS, T, 4, GSGN_MODE::PRECONDITIONER><<<grid, block, size_shared_memory>>>(
            data, M_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, nullptr
        );
    } else if(data.D == 2) {
        assert(data.M == 9);
        constexpr uint32_t bytes_per_gaussian = bytes_needed + sizeof(GaussianAttributeNoSH);
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_jt_kernel<GSGN_NUM_CHANNELS, T, 9, GSGN_MODE::PRECONDITIONER><<<grid, block, size_shared_memory>>>(
            data, M_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, nullptr
        );
    } else if(data.D == 3) {
        assert(data.M == 16);
        constexpr uint32_t bytes_per_gaussian = bytes_needed + sizeof(GaussianAttributeNoSH);
        uint32_t size_shared_memory = max_gaussians_per_block * bytes_per_gaussian;
        apply_jt_kernel<GSGN_NUM_CHANNELS, T, 16, GSGN_MODE::PRECONDITIONER><<<grid, block, size_shared_memory>>>(
            data, M_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, nullptr
        );
    }
}

template void CudaRasterizer::GSGN::calc_preconditioner<float>(PackedGSGNDataSpec& data, float* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);
template void CudaRasterizer::GSGN::calc_preconditioner<double>(PackedGSGNDataSpec& data, double* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);

//sort_sparse_jacobians 
template<typename T> void CudaRasterizer::GSGN::sort_sparse_jacobians(PackedGSGNDataSpec& data, T** in_sparse_jacobians, T** out_sparse_jacobians, int64_t** indices) {
    dim3 grid = dim3((data.max_n_sparse_gaussians * sizeof(GradientCache) + 255) / 256, data.num_images, 1);
    dim3 block = dim3(256, 1, 1);
    
    gsgn_sort_sparse_jacobians_kernel<GSGN_NUM_CHANNELS, T><<<grid, block>>>(
        data, in_sparse_jacobians, out_sparse_jacobians, indices
    );
}
template void CudaRasterizer::GSGN::sort_sparse_jacobians<float>(PackedGSGNDataSpec& data, float** in_sparse_jacobians, float** out_sparse_jacobians, int64_t** indices);
template void CudaRasterizer::GSGN::sort_sparse_jacobians<double>(PackedGSGNDataSpec& data, double** in_sparse_jacobians, double** out_sparse_jacobians, int64_t** indices);
template void CudaRasterizer::GSGN::sort_sparse_jacobians<__half>(PackedGSGNDataSpec& data, __half** in_sparse_jacobians, __half** out_sparse_jacobians, int64_t** indices);
