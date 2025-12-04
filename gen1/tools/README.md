# EgoBlur VRS Utilities

Meta has developed [EgoBlur](https://www.projectaria.com/tools/egoblur/), a sophisticated face and license plate anonymization system, as part of its ongoing commitment to responsible innovation. Since 2020, EgoBlur has been employed internally within the [Project Aria](https://www.projectaria.com/) program. The VRS file format is designed to record and play back streams of sensor data, including images, audio samples, and other discrete sensors (e.g., IMU, temperature), stored as time-stamped records within per-device streams. This library enables users to process VRS formatted videos using anonymization techniques and generate anonymized output videos in the same efficient VRS format.


# Getting Started

To use anonymization models with the VRS files, you need to download model files from our website and install necessary software as described below.


# Instruction to retrieve the ML Models

Models can be retrieved from [egoblur download](https://www.projectaria.com/tools/egoblur/) section. We will use the downloadable link and will fetch the models using “wget”.

We begin by creating a directory to store models

```
    mkdir ~/models && cd ~/models
```

Run command to fetch models
```
    wget -O face.zip "<downloadable_link_fetched_from_website>"
    unzip face.zip
```

Repeat the same process for the license plate model.


# Installation (Ubuntu 22.04 with libtorch 2.1 with CUDA toolkit 12.1)

This installation guide assumes that the nvidia-driver, Cuda, OpenCV, make and gcc are already installed. If not please follow the instructions at the end of this document to install these utilities/drivers.


## Download CMake

Download CMake binary

```
    mkdir ~/cmake && cd ~/cmake
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc4/cmake-3.28.0-rc4-linux-x86_64.sh
```

Unpack CMake

```
    chmod 555 cmake-3.28.0-rc4-linux-x86_64.sh && ./cmake-3.28.0-rc4-linux-x86_64.sh --skip-license
```


## Download libtorch

We are working with libtorch 2.1 with CUDA toolkit 12.1. This can be downloaded using:

```
cd ~/ && \
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip && \
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

## Install VRS dependencies

```
sudo apt install libfmt-dev libturbojpeg-dev libpng-dev && \
sudo apt install liblz4-dev libzstd-dev libxxhash-dev && \
sudo apt install libboost-system-dev libboost-iostreams-dev libboost-filesystem-dev libboost-thread-dev libboost-chrono-dev libboost-date-time-dev
```

## Install ninja build(required by projectaria_tools)

```
sudo apt install ninja-build
```

## Download github repositories

Make directory to hold repos

```
    mkdir ~/repos && cd ~/repos
```


### Torchvision

#### Download torchvision

```
    cd ~/repos && \
    git clone --branch v0.16.0 https://github.com/pytorch/vision/
```

#### Build torchvision

```
    cd ~/repos && \
    rm -rf vision/build && \
    mkdir vision/build && \
    cd vision/build && \
    ~/cmake/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST -DWITH_CUDA=on -DTorch_DIR=~/libtorch/share/cmake/Torch && \
    make -j && \
    sudo make install
```

### EgoBlur


#### Download EgoBlur repo

```
    cd ~/repos && \
    git clone https://github.com/facebookresearch/EgoBlur.git
```


#### Build ego_blur_vrs_mutation

```
    cd ~/repos/EgoBlur/tools/vrs_mutation && \
    rm -rf build && \
    mkdir build && \
    cd build &&  \
    ~/cmake/bin/cmake .. -DTorch_DIR=/home/$USER/libtorch/share/cmake/Torch -DTorchVision_DIR=~/repos/vision/cmake && \
    make -j ego_blur_vrs_mutation
```

# Usage:

## CLI Arguments

```
    -i,--in
```
use this argument to provide an absolute path for the given input VRS file on which we want to make detections and perform blurring. You MUST provide this value.

```
    -o,--out
```
 use this argument to provide an absolute path where we want to store the blurred VRS file. You MUST provide this value.

```
    -f, --faceModelPath
```
use this argument to provide an absolute EgoBlur face model file path. You SHOULD provide either --faceModelPath or --licensePlateModelPath or both. If none is provided code will not blur any data and the same input VRS will be written out without any operation.

```
    --face-model-confidence-threshold
```
use this argument to provide a face model score threshold to filter out low confidence face detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.


```
    -l, --licensePlateModelPath
```
use this argument to provide an absolute EgoBlur license plate model file path. You SHOULD provide either --faceModelPath or --licensePlateModelPath or both. If none is provided code will not blur any data and the same input VRS will be written out without any operation.

```
    --license-plate-model-confidence-threshold
```
use this argument to provide license plate model score threshold to filter out low confidence license plate detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

```
    --scale-factor-detections
```
use this argument to provide scale detections by the given factor to allow blurring more area. The values can only be positive real numbers eg: 0.9(values &lt; 1) would mean scaling DOWN the predicted blurred region by 10%, whereas as 1.1(values > 1) would mean scaling UP the predicted blurred region by 10%. If not provided this defaults to 1.15.

```
    --nms-threshold
```
use this argument to provide NMS iou threshold to filter out low confidence overlapping boxes. The values must be between 0.0 and 1.0, if not provided this defaults to 0.3.

```
    --use-gpu
```
flag to indicate whether you want to use GPU. It's highly recommended that you use GPU

A sample command using mandatory args only:
```
    cd ~/repos/EgoBlur/tools/vrs_mutation/build && \
    ./ego_blur_vrs_mutation --in your_vrs_file --out your_output_vrs_file -f ~/models/ego_blur_face.jit -l ~/models/    ego_blur_lp.jit --use-gpu
```

A sample command using all args:
```
cd ~/repos/EgoBlur/tools/vrs_mutation/build && \
./ego_blur_vrs_mutation --in your_vrs_file --out your_output_vrs_file -f ~/models/ego_blur_face.jit --face-model-confidence-threshold 0.75 -l ~/models/ego_blur_lp.jit --license-plate-model-confidence-threshold 0.99 --scale-factor-detections 1.15 --nms-threshold 0.3 --use-gpu
```

# Additional Installation Instructions

In this section we will cover additional installation instructions.


## Check OS version

```
    hostnamectl
```

This should give you the OS version which will be helpful in selecting the drivers and CUDA toolkit in the steps below.


## Install make
```
    sudo apt install make
```

## Install gcc
```
    sudo apt install gcc
```

## Install OpenCV
```
    sudo apt install libopencv-dev
```

## Install utility unzip
```
    sudo apt install unzip
```

## Check if you have GPU

This should provide the type of the GPU on the machine.

```
    lspci | grep nvidia -i
```

## Decide GPU Driver

Based on the gpu type and your OS type obtained previously, go to the website to search for an appropriate driver: [https://www.nvidia.com/Download/index.aspx?lang=en-us](https://www.nvidia.com/Download/index.aspx?lang=en-us)


## Update package manager
```
    sudo apt update
```

## Install GPU drivers
```
    sudo apt install ubuntu-drivers-common
```

Confirm that package manager identifies your device(GPU) and recommends appropriate drivers
```
    sudo ubuntu-drivers devices
```

Finally install the driver
```
    sudo apt install nvidia-driver-535
```

Reboot the system
```
    sudo reboot
```

Check if driver installation went correctly by running nvidia-smi
```
    nvidia-smi
```

## Install CUDA Toolkit

Go to pytorch website and find the specific cuda toolkit version you want to install(this should match with your libtorch supported version). Since we are using libtorch version v2.1.0: [Previous PyTorch Versions | PyTorch](https://fburl.com/himtgbgc) we will install libtorch v2.1.0: [Previous PyTorch Versions | PyTorch](https://fburl.com/z2m3p81z) with cuda toolkit 12.1.


### CUDA

Since we will be using libtorch v2.1.0 with cuda toolkit 12.1, visit website [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive) to get the installation instructions

Get cuda run file

```
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
```

Execute CUDA runfile

```
    sudo sh cuda_12.1.1_530.30.02_linux.run
```

Since we have already installed driver we don't need to reinstall it, we can simply continue with cuda toolkit installation

Export paths

```
    vi ~/.bashrc
```

And add these lines:

```
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Run
```
    source ~/.bashrc
```


To verify cuda toolkit installation run:
```
    nvcc --version
```
