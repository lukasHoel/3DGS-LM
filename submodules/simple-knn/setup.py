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
import os
import shutil
import torch

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

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
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu",
            "simple_knn.cu",
            ext_wrapper],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
