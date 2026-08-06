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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define GSGN_BLOCK_SIZE (GSGN_BLOCK_X * GSGN_BLOCK_Y)
#define GSGN_NUM_WARPS (GSGN_BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / GSGN_BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / GSGN_BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + GSGN_BLOCK_X - 1) / GSGN_BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + GSGN_BLOCK_Y - 1) / GSGN_BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ double3 transformPoint4x3(const double3& p, const float* matrix)
{
	double3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ double4 transformPoint4x4(const double3& p, const float* matrix)
{
	double4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ double3 transformVec4x3Transpose(const double3& p, const float* matrix)
{
	double3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float x, const float y, const float z, const float* matrix)
{
	float3 transformed = {
		matrix[0] * x + matrix[1] * y + matrix[2] * z,
		matrix[4] * x + matrix[5] * y + matrix[6] * z,
		matrix[8] * x + matrix[9] * y + matrix[10] * z,
	};
	return transformed;
}

__forceinline__ __device__ double3 transformVec4x3Transpose(const double x, const double y, const double z, const float* matrix)
{
	double3 transformed = {
		matrix[0] * x + matrix[1] * y + matrix[2] * z,
		matrix[4] * x + matrix[5] * y + matrix[6] * z,
		matrix[8] * x + matrix[9] * y + matrix[10] * z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 dnormvdv(float vx, float vy, float vz, float dvx, float dvy, float dvz)
{
	float sum2 = vx * vx + vy * vy + vz * vz;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - vx * vx) * dvx - vy * vx * dvy - vz * vx * dvz) * invsum32;
	dnormvdv.y = (-vx * vy * dvx + (sum2 - vy * vy) * dvy - vz * vy * dvz) * invsum32;
	dnormvdv.z = (-vx * vz * dvx - vy * vz * dvy + (sum2 - vz * vz) * dvz) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ double3 dnormvdv(double3 v, double3 dv)
{
	double sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	double invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	double3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 normv(float4 v)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    float inv_sqrt_sum2 = 1.0f / sqrt(sum2);
    float4 normv = { v.x * inv_sqrt_sum2, v.y * inv_sqrt_sum2, v.z * inv_sqrt_sum2, v.w * inv_sqrt_sum2 };
	return normv;
}

__forceinline__ __device__ float4 normv(float* v)
{
	float sum2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    float inv_sqrt_sum2 = 1.0f / sqrt(sum2);
    float4 normv = { v[0] * inv_sqrt_sum2, v[1] * inv_sqrt_sum2, v[2] * inv_sqrt_sum2, v[3] * inv_sqrt_sum2 };
	return normv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float* v, float4 dv)
{
	float sum2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v[0] * dv.x, v[1] * dv.y, v[2] * dv.z, v[3] * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v[0] * v[0]) * dv.x - v[0] * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v[1] * v[1]) * dv.y - v[1] * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v[2] * v[2]) * dv.z - v[2] * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v[3] * v[3]) * dv.w - v[3] * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float vx, float vy, float vz, float vw, float dvx, float dvy, float dvz, float dvw)
{
	float sum2 = vx * vx + vy * vy + vz * vz + vw * vw;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { vx * dvx, vy * dvy, vz * dvz, vw * dvw };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - vx * vx) * dvx - vx * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - vy * vy) * dvy - vy * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - vz * vz) * dvz - vz * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - vw * vw) * dvw - vw * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ double4 dnormvdv(double4 v, double4 dv)
{
	double sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	double invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	double4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	double vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	double4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ double sigmoid(double x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float dsigmoidvdv(float v, float dv)
{
	return sigmoid(v) * (1 - sigmoid(v)) * dv;
}

__forceinline__ __device__ double dsigmoidvdv(double v, double dv)
{
	return sigmoid(v) * (1 - sigmoid(v)) * dv;
}

__forceinline__ __device__ float dexpvdv(float v, float dv)
{
	return expf(v) * dv;
}

__forceinline__ __device__ double dexpvdv(double v, double dv)
{
	return exp(v) * dv;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
#if defined(USE_ROCM)
			__builtin_trap();  // HIP does not provide CUDA's __trap()
#else
			__trap();
#endif
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif