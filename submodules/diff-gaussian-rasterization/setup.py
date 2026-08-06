#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os
import shutil
os.path.dirname(os.path.abspath(__file__))

glm_include = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")

# --generate-line-info/-lineinfo/--use_fast_math are nvcc-only; hipcc rejects
# them. The wave64 reduction fixes (USE_ROCM-guarded) make the kernels correct
# without fast-math; the validation gate is loss-down / PSNR-up, not bit-exactness.
if torch.version.hip is not None:
    nvcc_flags = [glm_include]
else:
    nvcc_flags = [glm_include, "--generate-line-info", "-lineinfo", "--use_fast_math"]

# On Windows with HIP, MSVC-compiled ext.cpp cannot link c10::ValueError from
# the clang-built c10.dll (inherited ctor absent from the import lib). Rename
# ext.cpp to ext_winhip.cu so BuildExtension routes it through hipcc/amdclang++,
# which uses the same ABI as c10.dll.
if os.name == 'nt' and torch.version.hip is not None:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext.cpp")
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext_winhip.cu")
    shutil.copy2(src, dst)
    ext_wrapper = "ext_winhip.cu"
else:
    ext_wrapper = "ext.cpp"

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/gsgn.cu",
                "rasterize_points.cu",
                ext_wrapper
            ],
            extra_compile_args={"nvcc": nvcc_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
