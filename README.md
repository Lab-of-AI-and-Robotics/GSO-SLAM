<div align=center>

# GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry

[Jiung Yeon*](https://humdrum-balance-b8f.notion.site/Jiung-Yeon-6754922a22814c9a95af88801a96fb4b), [Seongbo Ha*](https://riboha.github.io), [Hyeonwoo Yu](https://bogus2000.github.io/)

(* Equal Contribution)

<h3 align="center"> IEEE Robotics and Automation Letters, 2026 </h3>

[Paper](https://arxiv.org/pdf/2602.11714) | [Video](https://www.youtube.com/watch?v=io5RmjNzkik&t)

</div>

## Environments
```bash
sudo apt-get update && sudo apt-get install -y \
    libsuitesparse-dev libeigen3-dev libboost-all-dev \
    libglm-dev libjsoncpp-dev libvtk9-dev libpcl-dev \
    libopencv-dev libglfw3-dev libglew-dev libzip-dev libflann-dev
```
Install [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [PyTorch](https://pytorch.org/get-started/locally/) respectively.
We used CUDA Version: 11.8 and PyTorch Version: 2.2.2


## Datasets
-Download
```bash
cd dataset
bash download_replica.sh
bash download_tum.sh
```
-Preprocess
```bash
bash preprocess.sh
```

## Run
### Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```
### Replica
```bash
cd experiments_bash
bash replica.sh
```
### TUM-RGBD
```bash
cd experiments_bash
bash tum.sh
```
If use_gaussian_viewer is set as 1, gaussian viewer will appear.

## Evaluation
```bash
cd experiments_bash
bash replica_eval_rendering.sh
bash replica_eval_depth.sh
```
