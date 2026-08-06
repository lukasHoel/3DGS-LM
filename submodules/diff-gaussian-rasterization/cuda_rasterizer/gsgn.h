#ifndef CUDA_RASTERIZER_GSGN_H_INCLUDED
#define CUDA_RASTERIZER_GSGN_H_INCLUDED

#include <vector>
#include <cuda.h>
#include "cuda_runtime.h"
#if !defined(USE_ROCM)
#include "device_launch_parameters.h"  // CUDA-toolkit IntelliSense header, no ROCm equivalent
#endif
#include "gsgn_data_spec.h"

// constants that identify the specialization of the generic_backward_kernel
enum class GSGN_MODE {
    EVAL_JTF_AND_SPARSE_INTERMEDIATE,
    PRECONDITIONER,
    APPLY_JTJ,
    APPLY_J
};

namespace CudaRasterizer
{
    namespace GSGN
    {

        void fill_reordered_geometry_buffer(const int P, const bool* clamped, const int* radii, const float2* means2D, const float* cov3Ds, const float4* conic_opacity, const float* colors, char* out_ptr);

        void filter_reordered_geometry_buffer(const int num_visible_gaussians, int* map_cache_to_gaussians, char* geom_buffer, char* out_geom_buffer);

        template<typename T> void eval_jtf_and_get_sparse_jacobian(PackedGSGNDataSpec& data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);

        template<typename T> void apply_j(PackedGSGNDataSpec& data, T* x_vec, T* x_resorted_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

        template<typename T> void apply_jt(PackedGSGNDataSpec& data, T* g_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

        template<typename T> void calc_preconditioner(PackedGSGNDataSpec& data, T* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);

        template<typename T> void sort_sparse_jacobians(PackedGSGNDataSpec& data, T** in_sparse_jacobians, T** out_sparse_jacobians, int64_t** indices);
    }
}

#endif