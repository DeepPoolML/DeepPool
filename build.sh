#!/bin/bash

set -x
set -e

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

# assumes pytorch installed in anaconda environment
export Torch_DIR=$CONDA_PREFIX/lib/python3.9/site-packages/torch/share/cmake/Torch/


export CUDAToolkit_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

pushd bench
python setup.py install &
setuppid=$!
popd

# CPP runtime build.
mkdir -p csrc/build
pushd csrc/build/
cmake ..
make -j
popd

wait $setuppid