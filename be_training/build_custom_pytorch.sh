#!/bin/bash

set -e
set -x

git clone https://github.com/pytorch/pytorch pytorch

cd pytorch
patch -p1 < ../pytorch.patch
git checkout ce05b7a3244ae7a61e989c9cd4eabf6d668ecbb0

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CUDA_HOME=/usr/local/cuda-11.0
export CUDNN_HOME=$CUDA_HOME

#export NCCL_ROOT_DIR=/home/ubuntu/nccl/build/
#export NCCL_LIB_DIR=/home/ubuntu/nccl/build/lib/
#export NCCL_INCLUDE_DIR=/home/ubuntu/nccl/build/include/
#export NCCL_VERSION=2.9.6

#export USE_SYSTEM_NCCL=1 
export USE_CUDNN=1
export REL_WITH_DEB_INFO=1
export CUDNN_ROOT_DIR=$CUDNN_HOME
export CUDNN_LIBRARY_PATH=$CUDNN_HOME/lib/
export CUDNN_INCLUDE_PATH=$CUDNN_HOME/include/

python setup.py install