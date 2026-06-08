# 3DGS-LM
3DGS-LM accelerates Gaussian-Splatting optimization by replacing the ADAM optimizer with Levenberg-Marquardt.

This is the official repository that contains source code for the ICCV 2025 paper [3DGS-LM](https://lukashoel.github.io/3DGS-LM).

[[arXiv](https://arxiv.org/abs/2409.12892)] [[Project Page](https://lukashoel.github.io/3DGS-LM/)] [[Video](https://youtu.be/tDiGuGMssg8)]

![Teaser](data/teaser.jpg "3DGS-LM")

If you find 3DGS-LM useful for your work please cite:
```
@InProceedings{hoellein_2025_3dgslm,
    title={3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt},
    author={H{\"o}llein, Lukas and Bo\v{z}i\v{c}, Alja\v{z} and Zollh{\"o}fer, Michael and Nie{\ss}ner, Matthias},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## Setup
We build our method on top of the original 3DGS source code.
Setup of the environment is identical. Please see the original repository for detailed instructions how to set it up: https://github.com/graphdeco-inria/gaussian-splatting.
This code release contains the submodule repositories ```diff-gaussian-rasterization``` and ```simple-knn```, so there is no need to clone them from the original source code.
In particular, the ```diff-gaussian-rasterization``` module is modified and contains the implementation of our Jacobian-vector product CUDA kernels.

### Building for AMD GPUs (ROCm/HIP)
The two CUDA extensions (`diff-gaussian-rasterization` and `simple-knn`) also build on AMD GPUs with a ROCm build of PyTorch -- they compile through PyTorch's build-time HIP translation, so the install is the same aside from setting the target architecture:
```
PYTORCH_ROCM_ARCH=gfx90a python -m pip install ./submodules/simple-knn --no-build-isolation --no-deps
PYTORCH_ROCM_ARCH=gfx90a python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
```
Set `PYTORCH_ROCM_ARCH` to your GPU architecture (for example `gfx90a` for CDNA2 / MI200, or `gfx1100` for RDNA3). The matrix-free PCG solver is pure PyTorch and runs unchanged.

## How to fit a scene with 3DGS-LM?
The script ```train.py``` is the main entry point to train a scene with our 3DGS-LM method. The ```arguments/__init__.py``` file contains additional command line arguments to run our method. One example how to execute our method on the ```garden``` scene from the ```Mip-NeRF 360``` dataset:

```
python train.py \
-s /path/to/360_v2/garden \
--root_out outputs/garden \
--exp_name "3dgs_lm_garden" \
--eval \
--images "images_4" \
--resolution 1 \
--image_subsample_size 25 \
--image_subsample_n_iters 4 \
--image_subsample_frame_selection_mode "strided" \
--num_sgd_iterations_before_gn 20000 \
--perc_images_in_line_search 0.3 \
--pcg_rtol 5e-2 \
--pcg_max_iter 8 \
--min_trust_region_radius 1e-4 \
--trust_region_radius 1e-3 \
--max_trust_region_radius 1e-2 \
--iterations 5 \
--test_iterations 5 \
--save_iterations 5
```

After training is completed, we automatically start rendering and evaluation scripts that calculate PSNR, SSIM, LPIPS metrics on the test set.
The results are saved to files in the output directory.
The runtime/memory consumption during training is saved in the same output directory to the files ```train_stats_<iteration>.json```, e.g. the file ```train_stats_5.json``` contains the argument ```ellipse_time``` which specifies the total runtime in seconds after 5 LM iterations.

## Reproduce Paper Results

In order to reproduce the Tab.1/Tab.6 results (3DGS+Ours):
- Specify the paths to the 360_v2, db, tandt datasets in ```scripts/fit_all_scenes.sh```.
- Run ```bash scripts/fit_all_scenes.sh``` on an NVIDIA A100 80GB GPU.
- Results are stored in ```outputs/<scene>```

## Where is the implementation of the LM optimizer?
Here we provide pointers to the main parts in the code that implement the LM optimizer:

The file ```submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py``` contains python bindings to the c++/cuda extension.
In particular the following methods facilitate the LM implementation: 

- ```eval_jtf_and_get_sparse_jacobian```: creates the cache and calculates ```b = J^T F(x)```. It refers to the ```buildCache``` kernel in the paper.
- ```sort_sparse_jacobians```: re-sorts the cache by Gaussians and refers to the ```sortCacheByGaussians``` kernel in the paper.
- ```calc_preconditioner```: calculates the Jacobi preconditioner and refers to the ```diagJTJ``` kernel in the paper.
- ```apply_jtj```: calculates the matrix-vector product ```g = J^T J p```. Internally, we make calls to multiple kernels that refer to the ```applyJ```, ```applyJT```, and ```sortX``` kernels.

The implementation of the CUDA kernels can be found in ```submodules/diff-gaussian-rasterization/cuda_rasterizer/gsgn.cu```.
Notably, we define multiple structures ```GaussianCache*``` that store all information for the gradient cache in the file ```submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h```.

The method ```lm_step``` in the ```train.py``` script calls into the c++/cuda extension to perform one LM iteration as outlined in the paper.

