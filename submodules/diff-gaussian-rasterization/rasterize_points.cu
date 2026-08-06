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

#include <math.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdio>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/gsgn_data_spec.h"
#include "rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const float cx,
    const float cy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
    const bool prepare_for_gsgn_backward, // if yes, will write out stuff to n_contrib_vol_rend and is_gaussian_hit (necessary for gsgn_backward)
	const bool debug)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({GSGN_NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, int_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;

    torch::Tensor n_contrib_vol_rend;
    torch::Tensor is_gaussian_hit;

    if(P != 0) {
        int M = 0;
        if(sh.size(0) != 0) {
            M = sh.size(1);
        }

        if(prepare_for_gsgn_backward) {
            n_contrib_vol_rend = torch::zeros({prepare_for_gsgn_backward ? H : 0, prepare_for_gsgn_backward ? W : 0}, int_opts);
            is_gaussian_hit = torch::zeros({prepare_for_gsgn_backward ? P : 0}, means3D.options().dtype(torch::kBool));
        }

        const auto result_tuple = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P, degree, M,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(), 
            opacity.contiguous().data_ptr<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(), 
            viewmatrix.contiguous().data_ptr<float>(), 
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            cx,
            cy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            prepare_for_gsgn_backward ? n_contrib_vol_rend.contiguous().data_ptr<int>() : nullptr,
            prepare_for_gsgn_backward ? is_gaussian_hit.contiguous().data_ptr<bool>() : nullptr,
            radii.contiguous().data_ptr<int>(),
            debug
        );

        rendered = std::get<0>(result_tuple);

        // reduce binning state to only point list
        const int64_t num_bytes_binning_point_list = std::get<1>(result_tuple);
        binningBuffer.resize_({(long long)num_bytes_binning_point_list});

        if(prepare_for_gsgn_backward) {
            torch::Tensor geomBufferReduced = torch::empty({0}, options.device(device));
            std::function<char*(size_t)> geomReducedFunc = resizeFunctional(geomBufferReduced);
            CudaRasterizer::Rasterizer::reorder_geometry_buffer(
                P,
                reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
                radii.contiguous().data_ptr<int>(),
                geomReducedFunc
            );
            geomBuffer = geomBufferReduced;
        }
    }

    return std::make_tuple(rendered, n_contrib_vol_rend, is_gaussian_hit, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const float cx,
    const float cy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int M = 0;
    if(sh.size(0) != 0) {
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, GSGN_NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if(P != 0) {
        CudaRasterizer::Rasterizer::backward(P, degree, M, R,
        background.contiguous().data_ptr<float>(),
        W, H, 
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        scales.data_ptr<float>(),
        scale_modifier,
        rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        cx,
        cy,
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dmeans2D.contiguous().data_ptr<float>(),
        dL_dconic.contiguous().data_ptr<float>(),  
        dL_dopacity.contiguous().data_ptr<float>(),
        dL_dcolors.contiguous().data_ptr<float>(),
        dL_dmeans3D.contiguous().data_ptr<float>(),
        dL_dcov3D.contiguous().data_ptr<float>(),
        dL_dsh.contiguous().data_ptr<float>(),
        dL_dscales.contiguous().data_ptr<float>(),
        dL_drotations.contiguous().data_ptr<float>(),
        debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0) {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}

// -----------------------------------
// ADDITIONAL METHODS FOR GN SUPPORT |
// -----------------------------------

std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> EvalJTFAndGetSparseJacobian(GSGNDataSpec& data) {
    data.init();

    auto options = data.params.options();
    if(data.use_double_precision) {
        options = options.dtype(torch::kFloat64);
    }
    torch::Tensor r_vec = torch::zeros({data.total_params}, options);

    std::vector<torch::Tensor> sparse_jacobians;
    std::vector<torch::Tensor> index_maps;
    std::vector<torch::Tensor> per_gaussian_caches;
    int** index_maps_ptr;
    __half** sparse_jacobians_ptr;
    float** per_gaussian_caches_ptr;

    cudaMallocManaged((void**) &index_maps_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &sparse_jacobians_ptr, data.num_images * sizeof(__half*));
    cudaMallocManaged((void**) &per_gaussian_caches_ptr, data.num_images * sizeof(float*));

    int per_gaussian_sparse_jac_size = 4;
    int per_gaussian_cache_size = 41;
    for(int i=0; i < data.num_images; i++) {
        int n_elem = data.num_sparse_gaussians[i];

        // each jacobian is a one-dimensional tensor that we index in the kernel in a custom way
#if defined(USE_ROCM)
        // The sparse-Jacobian emit (eval_jtf_sparse_render_bkwd_kernel) reserves
        // n_contrib_vol_rend slots per warp but can write a few fewer at large
        // scale, leaving reserved-but-unwritten trailing slots. On NVIDIA the
        // allocator zero-fills torch::empty and an out-of-range geomBuffer read
        // is a benign in-chunk over-read; on ROCm those uninitialized slots make
        // the consumer read geomBuffer[map_visible[garbage]] and page-fault.
        // Pre-fill the reserved slots with a guaranteed-visible Gaussian
        // (map_cache_to_gaussians[i][0] -> geomBuffer index 0) and ray 0, with a
        // zeroed Jacobian below, so any unwritten slot is an in-bounds,
        // exactly-zero-contribution entry. The emit overwrites the real entries.
        torch::Tensor sparse_jac = torch::zeros({n_elem * per_gaussian_sparse_jac_size}, options.dtype(at::kHalf));
#else
        torch::Tensor sparse_jac = torch::empty({n_elem * per_gaussian_sparse_jac_size}, options.dtype(at::kHalf));
#endif
        sparse_jacobians.push_back(sparse_jac);
        sparse_jacobians_ptr[i] = (__half*) sparse_jac.contiguous().data_ptr<at::Half>();

        // the index_map maps each entry in the jacobian to its gaussian_id (== index in data.means3D etc.)
        torch::Tensor index_map = torch::empty({n_elem * 2}, options.dtype(torch::kInt32));
#if defined(USE_ROCM)
        {
            int first_vis_gid = data.map_cache_to_gaussians[i].index({0}).item<int>();
            index_map.slice(0, 0, n_elem).fill_(first_vis_gid);   // gaussian-id half -> a visible Gaussian
            index_map.slice(0, n_elem, 2 * n_elem).fill_(0);       // ray-id half -> ray 0 (in bounds)
        }
#endif
        index_maps.push_back(index_map);
        index_maps_ptr[i] = index_map.contiguous().data_ptr<int>();

        // the per_gaussian_cache saves some intermediate calculations in the backward pass that are independent of the ray (per-image-per-gaussian)
        torch::Tensor per_gaussian_cache = torch::empty({data.n_visible_gaussians[i] * per_gaussian_cache_size}, options.dtype(torch::kFloat32));
        per_gaussian_caches.push_back(per_gaussian_cache);
        per_gaussian_caches_ptr[i] = per_gaussian_cache.contiguous().data_ptr<float>();
    }

    if(data.P != 0) {
        if(data.use_double_precision) {       
            CudaRasterizer::Rasterizer::eval_jtf_and_get_sparse_jacobian<double>(
                data,
                r_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr
            );
        } else {
            CudaRasterizer::Rasterizer::eval_jtf_and_get_sparse_jacobian<float>(
                data,
                r_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr
            );
        }
    }

    data.free_pointer_memory();
    cudaFree(index_maps_ptr);
    cudaFree(sparse_jacobians_ptr);
    cudaFree(per_gaussian_caches_ptr);

    return std::make_tuple(r_vec, sparse_jacobians, index_maps, per_gaussian_caches);
}

torch::Tensor ApplyJTJ(
    GSGNDataSpec& data,
    torch::Tensor x_vec,
    torch::Tensor x_resorted_vec,
    std::vector<torch::Tensor> sparse_jacobians,
    std::vector<torch::Tensor> index_map,
    std::vector<torch::Tensor> per_gaussian_cache,
    std::vector<torch::Tensor> segments,
    std::vector<torch::Tensor> segments_to_gaussians,
    std::vector<torch::Tensor> num_gaussians_in_block,
    std::vector<torch::Tensor> block_offset_in_segments
){
    data.init();

    CHECK_INPUT(x_vec);
    CHECK_INPUT(x_resorted_vec);

    auto options = data.params.options();
    if(data.use_double_precision) {
        options = options.dtype(torch::kFloat64);
    }
    torch::Tensor g_vec = torch::zeros({data.total_params}, options);
    torch::Tensor jx_vec = torch::zeros({data.jx_stride * 3}, options);

    int** index_maps_ptr;
    __half** sparse_jacobians_ptr;
    float** per_gaussian_caches_ptr;
    cudaMallocManaged((void**) &index_maps_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &sparse_jacobians_ptr, data.num_images * sizeof(__half*));
    cudaMallocManaged((void**) &per_gaussian_caches_ptr, data.num_images * sizeof(float*));

    int** segments_ptr;
    int** segments_to_gaussians_ptr;
    int** num_gaussians_in_block_ptr;
    int** block_offset_in_segments_ptr;
    int* max_gaussians_per_block_per_image_ptr;
    cudaMallocManaged((void**) &segments_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &segments_to_gaussians_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &num_gaussians_in_block_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &block_offset_in_segments_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &max_gaussians_per_block_per_image_ptr, data.num_images * sizeof(int));
    int max_gaussians_per_block = 0;

    for(int i=0; i < data.num_images; i++) {
        CHECK_INPUT(sparse_jacobians[i]);
        CHECK_INPUT(index_map[i]);
        CHECK_INPUT(per_gaussian_cache[i]);
        assert(sparse_jacobians[i].dtype() == at::kHalf);
        index_maps_ptr[i] = index_map[i].data_ptr<int>();
        sparse_jacobians_ptr[i] = (__half*) sparse_jacobians[i].contiguous().data_ptr<at::Half>();
        per_gaussian_caches_ptr[i] = per_gaussian_cache[i].data_ptr<float>();
        assert(per_gaussian_cache[i].size(0) == data.n_visible_gaussians[i] * 41);

        CHECK_INPUT(segments[i]);
        CHECK_INPUT(num_gaussians_in_block[i]);
        CHECK_INPUT(block_offset_in_segments[i]);
        segments_ptr[i] = segments[i].data_ptr<int>();
        segments_to_gaussians_ptr[i] = segments_to_gaussians[i].data_ptr<int>();
        num_gaussians_in_block_ptr[i] = num_gaussians_in_block[i].data_ptr<int>();
        block_offset_in_segments_ptr[i] = block_offset_in_segments[i].data_ptr<int>();
        int max_gaussians_per_block_per_image = num_gaussians_in_block[i].max().item<int>();
        max_gaussians_per_block_per_image_ptr[i] = max_gaussians_per_block_per_image;
        max_gaussians_per_block = max(max_gaussians_per_block, max_gaussians_per_block_per_image);
    }

    if(data.P != 0) {
        if(data.use_double_precision) {
            assert(x_vec.dtype() == torch::kFloat64);
            assert(x_resorted_vec.dtype() == torch::kFloat64);

            CudaRasterizer::Rasterizer::apply_j<double>(
                data,
                x_vec.data_ptr<double>(),
                x_resorted_vec.data_ptr<double>(),
                jx_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );

            // if we have weights: correct math requires to multiply the weight in both apply_j and apply_jt kernels
            // however, we can instead just do it in one kernel and multiply jx_vec with weight^2 (see paper)
            // also we can fuse weight and weight_ssim into one
            // this implementation leads to _no_ additional (uncoalesced) global memory read in the kernel.
            // instead we only have to perform these vector*vector multiplications/additions that are way faster.
            if(data.have_weights || data.have_weights_ssim) {
                if(data.have_weights && data.have_weights_ssim) {
                    jx_vec *= (data.weights * data.weights + data.weights_ssim * data.weights_ssim);
                }
                else if(data.have_weights) {
                    jx_vec *= (data.weights * data.weights);
                }
                else if(data.have_weights_ssim) {
                    jx_vec *= (data.weights_ssim * data.weights_ssim);
                } else {
                    assert(false);
                }
            }

            CudaRasterizer::Rasterizer::apply_jt<double>(
                data,
                g_vec.contiguous().data_ptr<double>(),
                jx_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );
        } else {
            assert(x_vec.dtype() == torch::kFloat32);
            assert(x_resorted_vec.dtype() == torch::kFloat32);

            CudaRasterizer::Rasterizer::apply_j<float>(
                data,
                x_vec.data_ptr<float>(),
                x_resorted_vec.data_ptr<float>(),
                jx_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );

            // if we have weights: correct math requires to multiply the weight in both apply_j and apply_jt kernels
            // however, we can instead just do it in one kernel and multiply jx_vec with weight^2 (see paper)
            // also we can fuse weight and weight_ssim into one
            // this implementation leads to _no_ additional (uncoalesced) global memory read in the kernel.
            // instead we only have to perform these vector*vector multiplications/additions that are way faster.
            if(data.have_weights || data.have_weights_ssim) {
                if(data.have_weights && data.have_weights_ssim) {
                    jx_vec *= (data.weights * data.weights + data.weights_ssim * data.weights_ssim);
                }
                else if(data.have_weights) {
                    jx_vec *= (data.weights * data.weights);
                }
                else if(data.have_weights_ssim) {
                    jx_vec *= (data.weights_ssim * data.weights_ssim);
                } else {
                    assert(false);
                }
            }

            CudaRasterizer::Rasterizer::apply_jt<float>(
                data,
                g_vec.contiguous().data_ptr<float>(),
                jx_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );
        }
    }

    data.free_pointer_memory();
    cudaFree(index_maps_ptr);
    cudaFree(sparse_jacobians_ptr);
    cudaFree(per_gaussian_caches_ptr);
    cudaFree(segments_ptr);
    cudaFree(segments_to_gaussians_ptr);
    cudaFree(num_gaussians_in_block_ptr);
    cudaFree(block_offset_in_segments_ptr);
    cudaFree(max_gaussians_per_block_per_image_ptr);

    return g_vec;
}

torch::Tensor CalcPreconditioner(GSGNDataSpec& data, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> index_map, std::vector<torch::Tensor> per_gaussian_cache, std::vector<torch::Tensor> segments, std::vector<torch::Tensor> segments_to_gaussians, std::vector<torch::Tensor> num_gaussians_in_block, std::vector<torch::Tensor> block_offset_in_segments) {
    data.init();

    auto options = data.params.options();
    if(data.use_double_precision) {
        options = options.dtype(torch::kFloat64);
    }
    torch::Tensor M_vec = torch::zeros({data.total_params}, options);

    int** index_maps_ptr;
    __half** sparse_jacobians_ptr;
    float** per_gaussian_caches_ptr;
    cudaMallocManaged((void**) &index_maps_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &sparse_jacobians_ptr, data.num_images * sizeof(__half*));
    cudaMallocManaged((void**) &per_gaussian_caches_ptr, data.num_images * sizeof(float*));

    int** segments_ptr;
    int** segments_to_gaussians_ptr;
    int** num_gaussians_in_block_ptr;
    int** block_offset_in_segments_ptr;
    cudaMallocManaged((void**) &segments_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &segments_to_gaussians_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &num_gaussians_in_block_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &block_offset_in_segments_ptr, data.num_images * sizeof(int*));
    int max_gaussians_per_block = 0;

    for(int i=0; i < data.num_images; i++) {
        CHECK_INPUT(sparse_jacobians[i]);
        CHECK_INPUT(index_map[i]);
        CHECK_INPUT(per_gaussian_cache[i]);
        assert(sparse_jacobians[i].dtype() == at::kHalf);
        index_maps_ptr[i] = index_map[i].data_ptr<int>();
        sparse_jacobians_ptr[i] = (__half*) sparse_jacobians[i].contiguous().data_ptr<at::Half>();
        per_gaussian_caches_ptr[i] = per_gaussian_cache[i].data_ptr<float>();
        assert(per_gaussian_cache[i].size(0) == data.n_visible_gaussians[i] * 41);

        CHECK_INPUT(segments[i]);
        CHECK_INPUT(num_gaussians_in_block[i]);
        CHECK_INPUT(block_offset_in_segments[i]);
        segments_ptr[i] = segments[i].data_ptr<int>();
        segments_to_gaussians_ptr[i] = segments_to_gaussians[i].data_ptr<int>();
        num_gaussians_in_block_ptr[i] = num_gaussians_in_block[i].data_ptr<int>();
        block_offset_in_segments_ptr[i] = block_offset_in_segments[i].data_ptr<int>();
        max_gaussians_per_block = max(max_gaussians_per_block, num_gaussians_in_block[i].max().item<int>());
    }

    if(data.P != 0) {
        // add together the weights here instead of in the kernel --> one less uncoalesced global memory read in the kernel
        if(data.have_weights && data.have_weights_ssim) {
            data.weights += data.weights_ssim;
        }

        if(data.use_double_precision) {
            CudaRasterizer::Rasterizer::calc_preconditioner<double>(
                data,
                M_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block
            );
        } else {
            CudaRasterizer::Rasterizer::calc_preconditioner<float>(
                data,
                M_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block
            );
        }

        // undo the addition again
        if(data.have_weights && data.have_weights_ssim) {
            data.weights -= data.weights_ssim;
        }
    }

    data.free_pointer_memory();
    cudaFree(index_maps_ptr);
    cudaFree(sparse_jacobians_ptr);
    cudaFree(per_gaussian_caches_ptr);
    cudaFree(segments_ptr);
    cudaFree(segments_to_gaussians_ptr);
    cudaFree(num_gaussians_in_block_ptr);
    cudaFree(block_offset_in_segments_ptr);

    return M_vec;
}

torch::Tensor ApplyJ(GSGNDataSpec& data, torch::Tensor x_vec, torch::Tensor x_resorted_vec, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> index_map, std::vector<torch::Tensor> per_gaussian_cache, std::vector<torch::Tensor> segments, std::vector<torch::Tensor> segments_to_gaussians, std::vector<torch::Tensor> num_gaussians_in_block, std::vector<torch::Tensor> block_offset_in_segments) {
    data.init();

    CHECK_INPUT(x_vec);
    CHECK_INPUT(x_resorted_vec);

    auto options = data.params.options();
    if(data.use_double_precision) {
        options = options.dtype(torch::kFloat64);
    }
    torch::Tensor jx_vec = torch::zeros({data.jx_stride * 3}, options);

    int** index_maps_ptr;
    __half** sparse_jacobians_ptr;
    float** per_gaussian_caches_ptr;
    cudaMallocManaged((void**) &index_maps_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &sparse_jacobians_ptr, data.num_images * sizeof(__half*));
    cudaMallocManaged((void**) &per_gaussian_caches_ptr, data.num_images * sizeof(float*));

    int** segments_ptr;
    int** segments_to_gaussians_ptr;
    int** num_gaussians_in_block_ptr;
    int** block_offset_in_segments_ptr;
    int* max_gaussians_per_block_per_image_ptr;
    cudaMallocManaged((void**) &segments_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &segments_to_gaussians_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &num_gaussians_in_block_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &block_offset_in_segments_ptr, data.num_images * sizeof(int*));
    cudaMallocManaged((void**) &max_gaussians_per_block_per_image_ptr, data.num_images * sizeof(int));
    int max_gaussians_per_block = 0;

    for(int i=0; i < data.num_images; i++) {
        CHECK_INPUT(sparse_jacobians[i]);
        CHECK_INPUT(index_map[i]);
        CHECK_INPUT(per_gaussian_cache[i]);
        assert(sparse_jacobians[i].dtype() == at::kHalf);
        index_maps_ptr[i] = index_map[i].data_ptr<int>();
        sparse_jacobians_ptr[i] = (__half*) sparse_jacobians[i].contiguous().data_ptr<at::Half>();
        per_gaussian_caches_ptr[i] = per_gaussian_cache[i].data_ptr<float>();
        assert(per_gaussian_cache[i].size(0) == data.n_visible_gaussians[i] * 41);

        CHECK_INPUT(segments[i]);
        CHECK_INPUT(num_gaussians_in_block[i]);
        CHECK_INPUT(block_offset_in_segments[i]);
        segments_ptr[i] = segments[i].data_ptr<int>();
        segments_to_gaussians_ptr[i] = segments_to_gaussians[i].data_ptr<int>();
        num_gaussians_in_block_ptr[i] = num_gaussians_in_block[i].data_ptr<int>();
        block_offset_in_segments_ptr[i] = block_offset_in_segments[i].data_ptr<int>();
        int max_gaussians_per_block_per_image = num_gaussians_in_block[i].max().item<int>();
        max_gaussians_per_block_per_image_ptr[i] = max_gaussians_per_block_per_image;
        max_gaussians_per_block = max(max_gaussians_per_block, max_gaussians_per_block_per_image);
    }

    if(data.P != 0) {
        if(data.use_double_precision) {
            assert(x_vec.dtype() == torch::kFloat64);
            assert(x_resorted_vec.dtype() == torch::kFloat64);

            CudaRasterizer::Rasterizer::apply_j<double>(
                data,
                x_vec.data_ptr<double>(),
                x_resorted_vec.data_ptr<double>(),
                jx_vec.contiguous().data_ptr<double>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );
        } else {
            assert(x_vec.dtype() == torch::kFloat32);
            assert(x_resorted_vec.dtype() == torch::kFloat32);

            CudaRasterizer::Rasterizer::apply_j<float>(
                data,
                x_vec.data_ptr<float>(),
                x_resorted_vec.data_ptr<float>(),
                jx_vec.contiguous().data_ptr<float>(),
                sparse_jacobians_ptr,
                index_maps_ptr,
                per_gaussian_caches_ptr,
                segments_ptr,
                segments_to_gaussians_ptr,
                num_gaussians_in_block_ptr,
                block_offset_in_segments_ptr,
                max_gaussians_per_block,
                max_gaussians_per_block_per_image_ptr
            );
        }
    }

    data.free_pointer_memory();
    cudaFree(index_maps_ptr);
    cudaFree(sparse_jacobians_ptr);
    cudaFree(per_gaussian_caches_ptr);
    cudaFree(segments_ptr);
    cudaFree(segments_to_gaussians_ptr);
    cudaFree(num_gaussians_in_block_ptr);
    cudaFree(block_offset_in_segments_ptr);
    cudaFree(max_gaussians_per_block_per_image_ptr);

    return jx_vec;
}

std::vector<torch::Tensor> SortSparseJacobians(GSGNDataSpec& data, std::vector<torch::Tensor> sparse_jacobians, std::vector<torch::Tensor> indices) {
    data.init();

    auto options = sparse_jacobians[0].options();

    std::vector<torch::Tensor> sorted_sparse_jacobians;
    int64_t** indices_ptr;
    cudaMallocManaged((void**) &indices_ptr, data.num_images * sizeof(int64_t*));

    int per_gaussian_sparse_jac_size = 4;
    for(int i=0; i < data.num_images; i++) {
        CHECK_INPUT(sparse_jacobians[i]);
        CHECK_INPUT(indices[i]);
        assert(sparse_jacobians[i].dtype() == options.dtype());

        // each jacobian is a one-dimensional tensor that we index in the kernel in a custom way
        int n_elem = data.num_sparse_gaussians[i];
        sorted_sparse_jacobians.push_back(torch::empty({n_elem * per_gaussian_sparse_jac_size}, options));
        indices_ptr[i] = indices[i].contiguous().data_ptr<int64_t>();
    }

    if(data.P != 0) {
        if(options.dtype() == torch::kFloat64) {
            double** in_sparse_jacobians_ptr;
            double** out_sparse_jacobians_ptr;
            cudaMallocManaged((void**) &in_sparse_jacobians_ptr, data.num_images * sizeof(double*));
            cudaMallocManaged((void**) &out_sparse_jacobians_ptr, data.num_images * sizeof(double*));

            for(int i=0; i < data.num_images; i++) {
                in_sparse_jacobians_ptr[i] = sparse_jacobians[i].contiguous().data_ptr<double>();
                out_sparse_jacobians_ptr[i] = sorted_sparse_jacobians[i].contiguous().data_ptr<double>();
            }
        
            CudaRasterizer::Rasterizer::sort_sparse_jacobians<double>(
                data,
                in_sparse_jacobians_ptr,
                out_sparse_jacobians_ptr,
                indices_ptr
            );

            cudaDeviceSynchronize();
            cudaFree(in_sparse_jacobians_ptr);
            cudaFree(out_sparse_jacobians_ptr);

        } else if(options.dtype() == torch::kFloat32) {
            float** in_sparse_jacobians_ptr;
            float** out_sparse_jacobians_ptr;
            cudaMallocManaged((void**) &in_sparse_jacobians_ptr, data.num_images * sizeof(float*));
            cudaMallocManaged((void**) &out_sparse_jacobians_ptr, data.num_images * sizeof(float*));

            for(int i=0; i < data.num_images; i++) {
                in_sparse_jacobians_ptr[i] = sparse_jacobians[i].contiguous().data_ptr<float>();
                out_sparse_jacobians_ptr[i] = sorted_sparse_jacobians[i].contiguous().data_ptr<float>();
            }

            CudaRasterizer::Rasterizer::sort_sparse_jacobians<float>(
                data,
                in_sparse_jacobians_ptr,
                out_sparse_jacobians_ptr,
                indices_ptr
            );

            cudaDeviceSynchronize();
            cudaFree(in_sparse_jacobians_ptr);
            cudaFree(out_sparse_jacobians_ptr);

        } else if(options.dtype() == torch::kFloat16) {
            __half** in_sparse_jacobians_ptr;
            __half** out_sparse_jacobians_ptr;
            cudaMallocManaged((void**) &in_sparse_jacobians_ptr, data.num_images * sizeof(__half*));
            cudaMallocManaged((void**) &out_sparse_jacobians_ptr, data.num_images * sizeof(__half*));

            for(int i=0; i < data.num_images; i++) {
                in_sparse_jacobians_ptr[i] = (__half*) sparse_jacobians[i].contiguous().data_ptr<at::Half>();
                out_sparse_jacobians_ptr[i] = (__half*) sorted_sparse_jacobians[i].contiguous().data_ptr<at::Half>();
            }

            CudaRasterizer::Rasterizer::sort_sparse_jacobians<__half>(
                data,
                in_sparse_jacobians_ptr,
                out_sparse_jacobians_ptr,
                indices_ptr
            );

            cudaDeviceSynchronize();
            cudaFree(in_sparse_jacobians_ptr);
            cudaFree(out_sparse_jacobians_ptr);
        }
    }

    data.free_pointer_memory();
    cudaFree(indices_ptr);

    return sorted_sparse_jacobians;
}

std::vector<torch::Tensor> FilterReorderedGeometryBuffer(std::vector<torch::Tensor> geomBuffer, std::vector<torch::Tensor> map_cache_to_gaussians, std::vector<int> num_visible_gaussians) {
    assert(geomBuffer.size() == map_cache_to_gaussians.size());
    assert(geomBuffer.size() == num_visible_gaussians.size());
    
    std::vector<torch::Tensor> filtered_geometry_buffers;

    auto options = geomBuffer[0].options();

    for(int i = 0; i < geomBuffer.size(); i++) {
        CHECK_INPUT(geomBuffer[i]);
        CHECK_INPUT(map_cache_to_gaussians[i]);

        torch::Tensor geomBufferFiltered = torch::empty({0}, options);
        std::function<char*(size_t)> geomBufferFilteredFunc = resizeFunctional(geomBufferFiltered);
        CudaRasterizer::Rasterizer::filter_reordered_geometry_buffer(
            num_visible_gaussians[i],
            map_cache_to_gaussians[i].data_ptr<int>(),
            reinterpret_cast<char*>(geomBuffer[i].data_ptr()),
            geomBufferFilteredFunc
        );
        filtered_geometry_buffers.push_back(geomBufferFiltered);
    }

    return filtered_geometry_buffers;
}