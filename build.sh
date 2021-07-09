#!/bin/bash

set -x

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

# CPP runtime build.
pushd csrc/build/
cmake ..
make -j
popd