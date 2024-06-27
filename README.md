 # Point Cloud Compression Using Deep Neural Networks
Created by Yinhao Wang

## Introduction

As point clouds has been widely used in a variety of applications like AR/VR, 3D modeling, autonomous drivings and bio-medical imagery, the point cloud compression (PCC) has became a major problem needs to be solved owing to the vast amount of data needed to faithfully represent real-word sceneries. While two different point cloud compression algorithms, G-PCC and V-PCC has been published as standard by the Moving Picture Experts Group (MPEG) recently, there are still large amounts of explorations about the application of the newly emerged learning-based method in PCC task, which have demonstrated their ability to outperform the rule-based approaches. One main difficulty in processing 3D point data wit neural networks is to find a proper representation of the raw point positions, which usually depict a continuous surface of object but are locally disconnected and sparsely distributed. While the voxelized representation enables the usage of convolution networks but lacks efficiency due to the local sparsity and the point based representation use specific network architecture to consume the raw point positions directly but have difficulties when generalizing to large-scale point clouds, what we would like to focus on is a heterogeneous scheme, the GRASP-Net, which uses a down-sampled coarse point cloud to divide points into local residuals and then encodes the local points with point-based network followed with a sparse convolutional network on the coarse point clouds with lower sparsity. In this work, we explored the possibility to use diffusion based point residual decoder in the PCC architecture. We have also proposed a way to estimate the original point cloud from the coarse one with quadratic fitting, which can both been used in the initialization of the diffusion decoder and to build up a different residual encoder. The implemented models are tested on four different real point cloud data selected from the MPEG dataset with the metrics of point-to-point distance (D1) and point-to-plane distance (D2). Based on our experiments, the quadratic fitting are able to provide a good estimation of the original point cloud, especially at low bit rates, while the diffusion point residual decoder still suffers from some point collapse problems that may caused by our applied Chamfer distance (CD) loss. The proposed point residual encoder shows similar performance with GRASP-Net and outperforms it at low bit rates. This work is heavily based on the previous work of [GRASP-Net](https://github.com/InterDigitalInc/GRASP-Net)

## Installation

The same installation steps as GRASP-Net
* Python 3.6, PyTorch 1.7.0, and CUDA 10.1. For this configuration, please launch the installation script `install_torch-1.7.0+cu-10.1.sh` with the following command:
```bash
echo y | conda create -n grasp python=3.6 && conda activate grasp && ./install_torch-1.7.0+cu-10.1.sh
```
* Python 3.8, PyTorch 1.8.1, and CUDA 11.1. For this configuration, please launch the installation script `install_torch-1.8.1+cu-11.2.sh` with the following command:
```bash
echo y | conda create -n grasp python=3.8 && conda activate grasp && ./install_torch-1.8.1+cu-11.2.sh
```
It is *highly recommended* to check the installation scripts which describe the details of the necessary packages. Note that [torchac](https://github.com/fab-jul/torchac) is used for arithmetic coding and [plyfile](https://github.com/dranjan/python-plyfile) is used for the reading/writing of PLY files. These two packages are under GPL license. By replacing them with another library providing the same functionality, our implementation can still run.

After that, put the binary of [`tmc3`](https://github.com/MPEGGroup/mpeg-pcc-tmc13) (MPEG G-PCC) and `pc_error` (D1 & D2 computation tool used in the MPEG group) under the `third_party` folder. A publicly-available version of `pc_error` can be found [here](https://github.com/NJUVISION/PCGCv2/blob/master/pc_error_d). To use it for the benchmarking of GRASP-Net, please download and rename it to `pc_error`.

## Datasets
Create a `datasets` folder then put all the datasets below. One may create soft links to the existing datasets to save space.

### ModelNet40

This work uses ModelNet40 to train for the case of surface point clouds. The ModelNet40 data loader is built on top of the loader of PyTorch Geometric. For the first run, it will automatically download the ModelNet40 data under the `datasets` folder and preprocess it. 

### Surface Point Clouds

The test set of the surface point clouds should be organized as shown below. Note that the point clouds are selected according to the MPEG recommendation [w21696](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/139_OnLine/w21696.zip).
```bash
${ROOT_OF_THE_REPO}/datasets/cat1
                               ├──A
                               │  ├── soldier_viewdep_vox12.ply
                               │  ├── boxer_viewdep_vox12.ply
                               │  ├── Facade_00009_vox12.ply
                               │  ├── House_without_roof_00057_vox12.ply
                               │  ├── queen_0200.ply
                               │  ├── soldier_vox10_0690.ply
                               │  ├── Facade_00064_vox11.ply
                               │  ├── dancer_vox11_00000001.ply
                               │  ├── Thaidancer_viewdep_vox12.ply
                               │  ├── Shiva_00035_vox12.ply
                               │  ├── Egyptian_mask_vox12.ply
                               │  └── ULB_Unicorn_vox13.ply
                               └──B
                                  ├── Arco_Valentino_Dense_vox12.ply
                                  └── Staue_Klimt_vox12.ply
```
Note that the file names are case-sensitive. Users may also put other surface point clouds in the `cat1/A` or `cat1/B` folders for additional testing.

## Basic Usages

The core of the training and benchmarking code are put below the `pccai/pipelines` folder. They are called by their wrappers below the `experiments` folder. The basic way to launch experiments with PccAI is:
 ```bash
 ./scripts/run.sh ./scripts/[filename].sh [launcher] [GPU ID(s)]
 ```
where `launcher` can be `s` (slurm), `d` (direct, run in background) and `f` (direct, run in foreground). `GPU ID(s)` can be ignored when launched with slurm. The results (checkpoints, point cloud files, log, *etc.*) will be generated under the `results/[filename]` folder. Note that multi-GPU training/benchmarking is not supported by GRASP-Net.

### Training

Take the training on the Ford sequences as an example, one can directly run
 ```bash
./scripts/run.sh ./scripts/train_grasp_quad/train_grasp_quad_r01.sh f 0
 ```
which trains the multi-reference model of the first rate point. The trained model will be generated under the `results/train_grasp_quad_r01` folder.

To understand the meanings of the options in the scripts for benchmarking/training, refer to `pccai/utils/option_handler.py` for details.

 ### Benchmarking

To benchmark the performance on the Ford sequences, one can directly run
 ```bash
./scripts/run.sh ./scripts/bench_grasp_quad/bench_grasp_quad_r01.sh f 0
 ```
which benchmarks all the first rate point of multi-reference model on GPU #0.