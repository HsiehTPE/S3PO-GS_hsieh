

<p align="center">
  <h1 align="center"> Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps
  </h1>
  <p align="center">
     <a ><strong>Chong Cheng*</strong></a>
    ·
    <a ><strong>Sicheng Yu*</strong></a>
    ·
    <a ><strong>Zijian Wang</strong></a>
    ·
    <a ><strong>Yifan Zhou</strong></a>
    ·
    <a href="https://wanghao.tech//"><strong>Hao Wang✉</strong></a>
  </p>
  <p align="center">The Hong Kong University of Science and Technology (GuangZhou)</p>
  <p align="center">(* Equal Contribution)</p>

  <h3 align="center"> ICCV 2025</h3>

[[Project page](https://3dagentworld.github.io/S3PO-GS/)],[[arxiv](https://arxiv.org/abs/2507.03737)]  

⭐Please also check out our previous work [[ICRA 25] OpenGS-SLAM](https://3dagentworld.github.io/opengs-slam/), on which S3PO-GS is a further extension.

# Getting Started

## Installation

1. Clone S3PO-GS.

```bash
git clone https://github.com/3DAgentWorld/S3PO-GS.git --recursive
cd S3PO-GS
```

2. Setup the environment.

```bash
conda env create -f environment.yml
conda activate S3PO-GS
```

3. Compile submodules for Gaussian splatting

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

4. Compile the cuda kernels for RoPE (as in CroCo v2 and MASt3R).

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

Our test setup was:

- Ubuntu 20.04: `pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cudatoolkit=11.8`
- NVIDIA RTX A6000

## Checkpoints

You can download the *'<u>MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric</u>'* checkpoint from the [MASt3R](https://github.com/naver/mast3r) code repository, and save it to the 'checkpoints' folder.

Alternatively, download it directly using the following method:

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

Please note that you must agree to the MASt3R license when using it.

## Downloading Datasets

#### Waymo

The processed data for the nine Waymo segments can be downloaded via [baidu](https://pan.baidu.com/s/1I1rnB6B8k2d4wzcRMT6gjA?pwd=omcg ) or [google](https://drive.google.com/drive/folders/1xUyNuNzUtsvZIV_q5Qz9zIXMGoMbLuCr?usp=sharing).

Save data under the `datasets/waymo` directory.

#### KITTI

The processed sample data for KITTI-07 can be downloaded via [baidu](https://pan.baidu.com/s/1-AmfeS-UYUJ9-sFFhO86wQ?pwd=wn4i) or [google](https://drive.google.com/drive/folders/1myR-cY3rBQBoLFZbKko36xDF2qUawJyW?usp=sharing).

Save data under the `datasets/KITTI` directory.

The full KITTI dataset can be downloaded from [The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The specific sequences used in this work are listed in KITTI_sequence_list.txt.

#### DL3DV

The processed data for the three DL3DV scenes can be downloaded via [baidu](https://pan.baidu.com/s/1LWuCnzojV5M-nl0Xf3hKvg?pwd=gjh5) or [google](https://drive.google.com/drive/folders/11K6lnSkFFiiCuJ9KG7II2bt0O7nevl7K?usp=sharing).

Save data under the `datasets/dl3dv` directory.

## Run

```bash
## Waymo-405841
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/waymo/405841.yaml"

## KITTI-07
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/KITTI/07.yaml"

## DL3DV-2
CUDA_VISIBLE_DEVICES=0 python slam.py --config "configs/mono/dl3dv/2.yaml"
```

## Demo

- If you want to view the real-time interactive SLAM window, please change `Results-use_gui` in `base_config.yaml` to True.

- When running on an Ubuntu system, a GUI window will pop up.

## Run on other dataset

- Please organize your data format and modify the code in `utils/dataset.py`.

- Ground truth depth input interface is still retained in the code, although we didn't use it for SLAM.

# Acknowledgement

- This work is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting),  [MonoGS](https://github.com/muskie82/MonoGS),  and [MASt3R](https://github.com/naver/mast3r), thanks for these great works.

- For more details about Demo, please refer to [MonoGS](https://github.com/muskie82/MonoGS), as we are using its visualization code.

