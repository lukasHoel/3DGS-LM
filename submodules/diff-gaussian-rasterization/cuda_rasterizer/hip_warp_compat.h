/*
 * ROCm/HIP wave64 compatibility shims for the 3DGS-LM rasterizer + LM kernels.
 *
 * The CUDA path is byte-identical: everything here is guarded by USE_ROCM (set
 * automatically by torch's HIP CUDAExtension build). On CUDA this header
 * defines GSGN_LANE_MASK as a 32-bit type and FULL_LANE_MASK as 0xffffffff,
 * matching the original literals.
 *
 * Two distinct lane groupings appear in this codebase and they need different
 * treatment on a 64-lane CDNA wavefront:
 *
 *  (1) The render-backward / segmented-reduce kernels partition the thread
 *      block into LOGICAL 32-thread warps (lane_id = tid % 32, a [NUM_WARPS][32]
 *      shared layout, and a host-side 32-pixel "warp" tiling in
 *      diff_gaussian_rasterization/__init__.py). On wave64 the hardware
 *      wavefront holds TWO such logical warps, so every per-warp collective must
 *      stay confined to its own 32-lane group. We do this with width-32 shuffles
 *      and cub::Warp{Reduce,Scan}<T, 32> (pinned; hipCUB defaults the logical
 *      width to warpSize == 64). Confining to width 32 keeps the two groups
 *      independent and matches the CUDA 32-lane warp exactly.
 *
 *  (2) The DISTWAR atomred_vec leader-election (atomred_vec) recombines
 *      per-primitive gradients via atomicAdd, so warp granularity is flexible
 *      there. We let it span the full hardware wavefront on HIP (a 64-bit
 *      same-primitive mask, __popcll/__ffsll, 1ull<<lane), which is correct and
 *      uses fewer atomics; __match_any_sync naturally partitions lanes by value
 *      so two distinct primitives in one wavefront each get their own leader.
 */

#ifndef GSGN_HIP_WARP_COMPAT_H_INCLUDED
#define GSGN_HIP_WARP_COMPAT_H_INCLUDED

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)

// HIP's *_sync builtins static_assert(sizeof(mask) == 8): the 32-bit literal
// 0xffffffff fails to compile. The full participation mask is 64-bit.
typedef unsigned long long GSGN_LANE_MASK;
#define GSGN_FULL_LANE_MASK 0xffffffffffffffffULL
// Full-wavefront participation mask passed to the *_sync builtins. Width-32
// shuffles still confine to a logical warp via the width argument; the mask
// only names the participating (active) lanes, which span the wavefront.
#ifndef FULL_MASK
#define FULL_MASK GSGN_FULL_LANE_MASK
#endif

// Logical warp width that the kernels in group (1) assume. Pinned to 32 on both
// backends so the [NUM_WARPS][32] layout and the 32-pixel host tiling agree.
#define GSGN_LOGICAL_WARP_SIZE 32

// Wavefront-relative lane (0..63 on CDNA, 0..31 on RDNA/CUDA).
__device__ __forceinline__ unsigned gsgn_wave_lane() { return __lane_id(); }

// 64-bit population count / find-first-set for the wavefront-wide masks used by
// the DISTWAR leader election in group (2).
__device__ __forceinline__ int gsgn_popc(GSGN_LANE_MASK m) { return __popcll((unsigned long long)m); }
__device__ __forceinline__ int gsgn_ffs(GSGN_LANE_MASK m) { return __ffsll((long long)m); }
__device__ __forceinline__ GSGN_LANE_MASK gsgn_lane_bit(unsigned lane) { return 1ull << lane; }

// The 32 bits of the wavefront mask belonging to THIS thread's 32-lane logical
// warp (the butterfly variant of the render-backward, group (1)). On CDNA the
// hardware wavefront holds two logical warps; a "full logical warp" match is
// (mask & this group's range) == this group's range, and the reduction stays
// width-32 with each group's own lane 0 doing the atomicAdd -- matching CUDA's
// one-atomicAdd-per-32-warp granularity exactly.
__device__ __forceinline__ GSGN_LANE_MASK gsgn_logical_warp_mask() {
    return 0xffffffffull << (32u * (__lane_id() / 32u));
}

// Warp barrier for a site reached AFTER a per-thread-divergent `continue`, where
// the pre-divergence ballot no longer names exactly the surviving lanes. HIP's
// __syncwarp(MaskT) asserts mask == __ballot(true) (active under torch's .cu
// device compile, which carries no -DNDEBUG), so an over-broad mask traps with
// HSA_STATUS_ERROR_EXCEPTION code 0x1016. The maskless __syncwarp() is a
// wavefront barrier over exactly the lanes still executing -- a superset of the
// synchronization the following HeadSegmentedSum + head-flag atomicAdd needs.
#define GSGN_SYNCWARP_AFTER_DIVERGENCE(mask) __syncwarp()

// Participation mask for a *_sync shuffle reached under partial wavefront
// activity (the DISTWAR butterfly is entered per 32-lane group, so only one of
// the two groups in a wavefront may be live). HIP's __shfl_*_sync uses the mask
// ONLY for the mask == __ballot(true) assertion; the width argument does the
// 32-lane sub-grouping, independent of the mask. A fixed FULL_MASK (all 64)
// over-names the inactive group and traps, and a per-group half-mask also traps
// when BOTH groups are live (then __ballot(true) is all 64, not the half). The
// active set itself, __activemask() == __ballot(true), always satisfies the
// assertion while width=32 preserves the per-group reduction.
__device__ __forceinline__ GSGN_LANE_MASK gsgn_active_shfl_mask() { return __activemask(); }

#else  // CUDA: keep the original 32-bit spelling byte-for-byte.

typedef unsigned GSGN_LANE_MASK;
#define GSGN_FULL_LANE_MASK 0xffffffffu
#ifndef FULL_MASK
#define FULL_MASK GSGN_FULL_LANE_MASK
#endif
#define GSGN_LOGICAL_WARP_SIZE 32

__device__ __forceinline__ unsigned gsgn_wave_lane() {
    return (threadIdx.y * blockDim.x + threadIdx.x) & 31;
}
__device__ __forceinline__ int gsgn_popc(GSGN_LANE_MASK m) { return __popc(m); }
__device__ __forceinline__ int gsgn_ffs(GSGN_LANE_MASK m) { return __ffs(m); }
__device__ __forceinline__ GSGN_LANE_MASK gsgn_lane_bit(unsigned lane) { return 1u << lane; }
__device__ __forceinline__ GSGN_LANE_MASK gsgn_logical_warp_mask() { return 0xffffffffu; }

// On CUDA this is byte-identical to the upstream __syncwarp(mask): NVIDIA
// tolerates a mask naming a now-inactive lane.
#define GSGN_SYNCWARP_AFTER_DIVERGENCE(mask) __syncwarp(mask)

// On CUDA the butterfly shuffle keeps its original FULL_MASK (0xffffffff): the
// reduction is over a full 32-lane warp, byte-identical to upstream.
__device__ __forceinline__ GSGN_LANE_MASK gsgn_active_shfl_mask() { return FULL_MASK; }

#endif

#endif // GSGN_HIP_WARP_COMPAT_H_INCLUDED
