/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#if !defined(USE_ROCM)
#include "device_launch_parameters.h"  // CUDA-toolkit IntelliSense header, no ROCm equivalent; symbols are hipcc intrinsics
#endif
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#if !defined(USE_ROCM)
#include <cooperative_groups/reduce.h>  // HIP has no cg::reduce header; only this_grid/this_thread_block are used here
#endif
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "gsgn.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{			
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::BinningStateReduced CudaRasterizer::BinningStateReduced::fromChunk(char*& chunk, size_t P)
{
	BinningStateReduced binning;
	obtain(chunk, binning.point_list, P, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int, int64_t> CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
    const float cx, float cy,
	const bool prefiltered,
	float* out_color,
    int* n_contrib_vol_rend,
	bool* is_gaussian_hit,
    int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + GSGN_BLOCK_X - 1) / GSGN_BLOCK_X, (height + GSGN_BLOCK_Y - 1) / GSGN_BLOCK_Y, 1);
	dim3 block(GSGN_BLOCK_X, GSGN_BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (GSGN_NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
        cx, cy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // Create BinningState and also remember how many bytes are needed for point_list, because that is the only thing needed afterwards (for backward or GN)
    // we can then resize the tensor using this information and reduce the memory by 90%
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
    char* begin_binning_point_list = binning_chunkptr;
    char* end_binning_point_list = binning_chunkptr;
    BinningStateReduced::fromChunk(end_binning_point_list, num_rendered);
    int64_t num_bytes_binning_point_list = end_binning_point_list - begin_binning_point_list;
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
            binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
        binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
        n_contrib_vol_rend,
        is_gaussian_hit,
		background,
		out_color), debug)

	return std::make_tuple(num_rendered, num_bytes_binning_point_list);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
    const float cx, float cy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningStateReduced binningState = BinningStateReduced::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + GSGN_BLOCK_X - 1) / GSGN_BLOCK_X, (height + GSGN_BLOCK_Y - 1) / GSGN_BLOCK_Y, 1);
	const dim3 block(GSGN_BLOCK_X, GSGN_BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
        cx, cy,
        width, height,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}

// -----------------------------------
// ADDITIONAL METHODS FOR GN SUPPORT |
// -----------------------------------

void CudaRasterizer::PackedGSGNDataSpec::allocate_pointer_memory() {
    assert(! pointers_changed);

    cudaMallocManaged((void**) &point_list_ptrs, num_images * sizeof(uint32_t*));
    cudaMallocManaged((void**) &ranges_ptrs, num_images * sizeof(uint2*));
    cudaMallocManaged((void**) &n_contrib_ptrs, num_images * sizeof(int32_t*));
    cudaMallocManaged((void**) &accum_alpha_ptrs, num_images * sizeof(float*));

    for(int i=0; i < num_images; i++) {
        BinningStateReduced binningState = BinningStateReduced::fromChunk(binningBuffer_ptrs[i], num_rendered[i]);
        point_list_ptrs[i] = binningState.point_list;

        ImageState imgState = ImageState::fromChunk(imageBuffer_ptrs[i], W * H);
        ranges_ptrs[i] = imgState.ranges;
        n_contrib_ptrs[i] = imgState.n_contrib;
        accum_alpha_ptrs[i] = imgState.accum_alpha;
    }

    pointers_changed = true;
}

void CudaRasterizer::Rasterizer::reorder_geometry_buffer(const int P, char* geom_buffer, int* radii, std::function<char* (size_t)> out_geom_buffer) {
    // cast char* to GeometryState
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    if (radii == nullptr) {
		radii = geomState.internal_radii;
	}

    // resize required bytes for GeometryStateReduced
    // chunkptr is a GPU pointer to the start of the memory
    size_t chunk_size = P * sizeof(GeometryStateReduced);
	char* chunkptr = out_geom_buffer(chunk_size);

    // fill it
    CHECK_CUDA(CudaRasterizer::GSGN::fill_reordered_geometry_buffer(
        P,
        geomState.clamped,
        radii,
        geomState.means2D,
        geomState.cov3D,
        geomState.conic_opacity,
        geomState.rgb,
        chunkptr
    ), false)

}

void CudaRasterizer::Rasterizer::filter_reordered_geometry_buffer(const int num_visible_gaussians, int* map_cache_to_gaussians, char* geom_buffer, std::function<char* (size_t)> out_geom_buffer) {
    // resize required bytes for GeometryStateReduced
    // chunkptr is a GPU pointer to the start of the memory
    size_t chunk_size = num_visible_gaussians * sizeof(GeometryStateReduced);
	char* chunkptr = out_geom_buffer(chunk_size);

    // fill it
    CHECK_CUDA(CudaRasterizer::GSGN::filter_reordered_geometry_buffer(
        num_visible_gaussians,
        map_cache_to_gaussians,
        geom_buffer,
        chunkptr
    ), false)
}

// eval_jtf_and_get_sparse_jacobian
template<typename T> void CudaRasterizer::Rasterizer::eval_jtf_and_get_sparse_jacobian(PackedGSGNDataSpec data, T* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache) {
    data.allocate_pointer_memory();
	CHECK_CUDA(CudaRasterizer::GSGN::eval_jtf_and_get_sparse_jacobian<T>(data, r_vec, sparse_jacobians, index_map, per_gaussian_cache), data.debug)
    data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::eval_jtf_and_get_sparse_jacobian<float>(PackedGSGNDataSpec data, float* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);
template void CudaRasterizer::Rasterizer::eval_jtf_and_get_sparse_jacobian<double>(PackedGSGNDataSpec data, double* r_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache);

// apply_j
template<typename T> void CudaRasterizer::Rasterizer::apply_j(PackedGSGNDataSpec data, T* x_vec, T* x_resorted_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr) {
    // data.allocate_pointer_memory();
	CHECK_CUDA(CudaRasterizer::GSGN::apply_j<T>(data, x_vec, x_resorted_vec, jx_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, max_gaussians_per_block_per_image_ptr), data.debug)
    // data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::apply_j<float>(PackedGSGNDataSpec data, float* x_vec, float* x_resorted_vec, float* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);
template void CudaRasterizer::Rasterizer::apply_j<double>(PackedGSGNDataSpec data, double* x_vec, double* x_resorted_vec, double* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

// apply_jt
template<typename T> void CudaRasterizer::Rasterizer::apply_jt(PackedGSGNDataSpec data, T* g_vec, T* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr) {
    // data.allocate_pointer_memory();
	CHECK_CUDA(CudaRasterizer::GSGN::apply_jt<T>(data, g_vec, jx_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block, max_gaussians_per_block_per_image_ptr), data.debug)
    // data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::apply_jt<float>(PackedGSGNDataSpec data, float* g_vec, float* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);
template void CudaRasterizer::Rasterizer::apply_jt<double>(PackedGSGNDataSpec data, double* g_vec, double* jx_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block, int* max_gaussians_per_block_per_image_ptr);

// calc_preconditioner
template<typename T> void CudaRasterizer::Rasterizer::calc_preconditioner(PackedGSGNDataSpec data, T* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block) {
    // data.allocate_pointer_memory();
	CHECK_CUDA(CudaRasterizer::GSGN::calc_preconditioner<T>(data, M_vec, sparse_jacobians, index_map, per_gaussian_cache, segments, segments_to_gaussians, num_gaussians_in_block, block_offset_in_segments, max_gaussians_per_block), data.debug)
    // data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::calc_preconditioner(PackedGSGNDataSpec data, float* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);
template void CudaRasterizer::Rasterizer::calc_preconditioner(PackedGSGNDataSpec data, double* M_vec, __half** sparse_jacobians, int** index_map, float** per_gaussian_cache, int** segments, int** segments_to_gaussians, int** num_gaussians_in_block, int** block_offset_in_segments, int max_gaussians_per_block);

// sort_sparse_jacobians
template<typename T> void CudaRasterizer::Rasterizer::sort_sparse_jacobians(PackedGSGNDataSpec data, T** in_sparse_jacobians, T** out_sparse_jacobians, int64_t** indices) {
	// data.allocate_pointer_memory();
    CHECK_CUDA(CudaRasterizer::GSGN::sort_sparse_jacobians<T>(data, in_sparse_jacobians, out_sparse_jacobians, indices), data.debug)
    // data.free_pointer_memory();
}
template void CudaRasterizer::Rasterizer::sort_sparse_jacobians<float>(PackedGSGNDataSpec data, float** in_sparse_jacobians, float** out_sparse_jacobians, int64_t** indices);
template void CudaRasterizer::Rasterizer::sort_sparse_jacobians<double>(PackedGSGNDataSpec data, double** in_sparse_jacobians, double** out_sparse_jacobians, int64_t** indices);
template void CudaRasterizer::Rasterizer::sort_sparse_jacobians<__half>(PackedGSGNDataSpec data, __half** in_sparse_jacobians, __half** out_sparse_jacobians, int64_t** indices);
